"""
Interface for handling data from a star catalogue. Retrieve data as well as
render and write images.
"""

import copy
import math
import subprocess
import sys
import time
from pathlib import Path

import bpy
import numpy as np
import OpenEXR
import skimage.filters
import skimage.transform


class StarCatalogError(RuntimeError):
    """Generic error for star catalog module."""
    pass


class StarCatalog():
    """Class to access star catalogs and render stars."""

    def __init__(self, res_dir):
        """."""

        self.root_dir = Path(__file__).parent.parent.parent
        self.starcat_dir = self.root_dir / "data" / "UCAC4"
        self.res_dir = res_dir

        self.exe = self.root_dir / "software" / "star_cats" / "u4test.exe"

        self.cmd = f'"{str(self.exe)}" {{0}} {{1}} {{2}} {{3}} -h ' \
                   f'"{str(self.starcat_dir)}" "{{4}}"'

        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        # How to suppress crash notification dialog?, Jan 14,2004 -
        # Raymond Chen"s response [1]
        if sys.platform.startswith("win"):
            import ctypes
            SEM_NOGPFAULTERRORBOX = 0x0002  # From MSDN
            ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)

    def get_stardata(self, ra, dec, width, height, filename="ucac4.txt"):
        """Retrieve star data from given field of view using UCAC4 catalog."""
        res_file = self.res_dir / filename

        command = self.cmd.format(ra, dec, width, height, str(res_file))

        for _ in range(5):
            retcode = subprocess.call(command)

            if retcode > 0:
                break

        with open(str(res_file), "r") as rfile:
            complete_data = rfile.readlines()

        star_data = []
        for line in complete_data[1:]:
            line_data = line.split()

            ra_star = float(line_data[1])
            dec_star = float(line_data[2])
            mag_star = float(line_data[3])

            star_data.append((ra_star, dec_star, mag_star))

        return star_data

    @classmethod
    def create_starmap(cls, stardata, cam_direction, right_vec, up_vec, res_x, res_y, filename):
        """Create a starmap from given data and field of view."""
        up_vec -= cam_direction
        right_vec -= cam_direction
        total_flux = 0.

        f_over_h_ccd_2 = 1. / np.sqrt(np.dot(up_vec, up_vec))
        up_norm = up_vec * f_over_h_ccd_2
        f_over_w_ccd_2 = 1. / np.sqrt(np.dot(right_vec, right_vec))
        right_norm = right_vec * f_over_w_ccd_2

        print("F_over_w {} f_over_h {}".format(f_over_w_ccd_2, f_over_h_ccd_2))
        print("Res {} x {}".format(res_x, res_y))

        ss = 2
        starmap = np.zeros((res_y * ss, res_x * ss, 4), np.float32)
        
        for star_data in stardata:
            star_data = copy.copy(star_data)
            ra_star = math.radians(star_data[0])
            dec_star = math.radians(star_data[1])

            z_star = math.sin(dec_star)
            x_star = math.cos(dec_star) * math.cos(ra_star - math.pi)
            y_star = -math.cos(dec_star) * math.sin(ra_star - math.pi)
            vec = [x_star, y_star, z_star]
            vec2 = [x_star, -y_star, z_star]
            if np.dot(vec, cam_direction) < np.dot(vec2, cam_direction):
                vec = vec2

            x_pix = (f_over_w_ccd_2 * np.dot(right_norm, vec) / np.dot(cam_direction, vec) + 1.) * (res_x - 1) / 2.
            y_pix = (-f_over_h_ccd_2 * np.dot(up_norm, vec) / np.dot(cam_direction, vec) + 1.) * (res_y - 1) / 2.
            x_pix2 = max(0, min(int(round(x_pix * ss)), res_x * ss - 1))
            y_pix2 = max(0, min(int(round(y_pix * ss)), res_y * ss - 1))

            flux = math.pow(10., -0.4 * (star_data[2]))
            flux0 = math.pow(10., -0.4 * (star_data[2]))

            pix = starmap[y_pix2, x_pix2]
            starmap[y_pix2, x_pix2] = [pix[0] + flux,
                                       pix[1] + flux, pix[2] + flux, 1.]

            total_flux += flux0
        
        starmap_gaussian = skimage.filters.gaussian(starmap, ss / 2., multichannel=True)
        starmap_downscaled = np.zeros((res_y, res_x, 4), np.float32)
        for c in range(4):
                starmap_downscaled[:, :, c] = skimage.transform.downscale_local_mean(starmap_gaussian[:, :, c], (ss, ss)) * (ss * ss)
    
        cls.write_openexr_image(filename, starmap_downscaled)
    
        return (total_flux, np.sum(starmap_downscaled[:, :, 0]))
    
    @staticmethod
    def write_openexr_image(filename, picture):
        """Save image in OpenEXR file format."""
        filename = str(filename)
    
        file_extension = ".exr"
        if filename[-4:] != file_extension:
            filename += file_extension
    
        height = len(picture)
        width = len(picture[0])
        channels = len(picture[0][0])
    
        if channels == 4:
            data_r = picture[:, :, 0].tobytes()
            data_g = picture[:, :, 1].tobytes()
            data_b = picture[:, :, 2].tobytes()
            data_a = picture[:, :, 3].tobytes()
            image_data = {"R": data_r, "G": data_g, "B": data_b, "A": data_a}
        elif channels == 3:
            data_r = picture[:, :, 0].tobytes()
            data_g = picture[:, :, 1].tobytes()
            data_b = picture[:, :, 2].tobytes()
            image_data = {"R": data_r, "G": data_g, "B": data_b}
        else:
            raise StarCatalogError("Invalid number of channels of starmap image.")
        
        hdr = OpenEXR.Header(width, height)
        file_handler = OpenEXR.OutputFile(filename, hdr)
        file_handler.writePixels(image_data)
        file_handler.close()


#class StarCache:
#    """Handling stars in field of view, for rendering of scene."""
#
#    def __init__(self, template=None, parent=None):
#        """Initialise StarCache."""
#        self.template = template
#        self.star_array = []
#        self.parent = parent
#
#    def set_stars(self, stardata, star_template, cam_direction, sat_pos_rel, R, pixelsize_at_R, scene_names):
#        """Set current stars in the field of view."""
#        if len(self.star_array) < len(stardata):
#            for _ in range(0, len(stardata) - len(self.star_array)):
#                new_obj = self.template.copy()
#                new_obj.data = self.template.data.copy()
#                new_obj.animation_data_clear()
#                new_mat = star_template.material_slots[0].material.copy() # TODO: check for star_template use, changed to input for now
#                new_obj.material_slots[0].material = new_mat
#                self.star_array.append(new_obj)
#                if self.parent is not None:
#                    new_obj.parent = self.parent
#        total_flux = 0.
#
#        for i in range(0, len(stardata)):
#            star = self.star_array[i]
#            star_data = copy.copy(stardata[i])
#            star_data[0] = math.radians(star_data[0])
#            star_data[1] = math.radians(star_data[1])
#
#            z_star = math.sin(star_data[1])
#            x_star = math.cos(star_data[1]) * math.cos(star_data[0] - math.pi)
#            y_star = -math.cos(star_data[1]) * math.sin(star_data[0] - math.pi)
#            vec = [x_star, y_star, z_star]
#            vec2 = [x_star, -y_star, z_star]
#            if np.dot(vec, cam_direction) < np.dot(vec2, cam_direction):
#                vec = vec2
#
#            pixel_factor = 10
#            # Always keep satellite in center to emulate large distances
#            star.location = np.asarray(vec) * R + sat_pos_rel
#            star.scale = (pixelsize_at_R / pixel_factor, pixelsize_at_R / pixel_factor,
#                          pixelsize_at_R / pixel_factor)
#
#            flux = math.pow(10, -0.4 * (star_data[2] - 10.))
#            flux0 = math.pow(10, -0.4 * (star_data[2]))
#            total_flux += flux0
#
#            star.material_slots[0].material.node_tree.nodes.get("Emission").inputs[1].default_value = flux * pixel_factor * pixel_factor
#
#            for scene_name in scene_names:
#                scene = bpy.data.scenes[scene_name]
#                if star.name not in scene.objects:
#                    scene.objects.link(star)
#        print("{} stars set, buffer len {}".format(i, len(self.star_array)))
#        if len(self.star_array) > len(stardata):
#            for scene_name in scene_names:
#                scene = bpy.data.scenes[scene_name]
#                for i in range(len(stardata), len(self.star_array)):
#                    if self.star_array[i].name in scene.objects:
#                        scene.objects.unlink(self.star_array[i])
#
#        return total_flux
#
