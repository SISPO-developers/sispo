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

def get_ucac4(ra, ra_w, dec, dec_h, filename="ucac4.txt"):
    """Retrieve starmap data from UCAC4 catalog."""
    errorlog_fn = "starfield_errorlog%f.txt" % time.time()

    if sys.platform.startswith("win"):
        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        # How to suppress crash notification dialog?, Jan 14,2004 -
        # Raymond Chen"s response [1]

        import ctypes
        SEM_NOGPFAULTERRORBOX = 0x0002  # From MSDN
        ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)

    project_root = Path.cwd()
    ucac4 = project_root.joinpath("data").joinpath("UCAC4")
    u4test = project_root.joinpath("software").joinpath("star_cats").joinpath("u4test.exe")
    res_file = project_root / "data" / "results" / "Didymos" / filename
    res_file.resolve()

    command = '"' + str(u4test) + '" {} {} {} {}'.format(ra, dec, ra_w, dec_h) + ' -h "' + str(ucac4) + '" "{}"'.format(str(res_file))
    print(command)

    for _ in range(0, 5):
        retcode = subprocess.call(command)
        print("Retcode ", retcode)
        if retcode > 0:
            break
        with open(errorlog_fn, "at") as fout:
            fout.write("{},\'{}\',{}\n".format(time.time(), command, retcode))

    with open(str(res_file), "r") as file:
        lines = file.readlines()
        print("Lines", len(lines))
    out = []
    for line in lines[1:]:

        ra_star = float(line[11:23])
        dec_star = float(line[23:36])
        mag_star = float(line[36:43])
        out.append([ra_star, dec_star, mag_star])
    return out

class StarCache:
    """Handling stars in field of view, for rendering of scene."""

    def __init__(self, template, parent = None):
        """Initialise StarCache."""
        self.template = template
        self.star_array = []
        self.parent = parent

    def set_stars(self, stardata, star_template, cam_direction, sat_pos_rel, R, pixelsize_at_R, scene_names):
        """Set current stars in the field of view."""
        if len(self.star_array) < len(stardata):
            for _ in range(0, len(stardata) - len(self.star_array)):
                new_obj = self.template.copy()
                new_obj.data = self.template.data.copy()
                new_obj.animation_data_clear()
                new_mat = star_template.material_slots[0].material.copy() # TODO: check for star_template use, changed to input for now
                new_obj.material_slots[0].material = new_mat
                self.star_array.append(new_obj)
                if self.parent is not None:
                    new_obj.parent = self.parent
        total_flux = 0.

        for i in range(0, len(stardata)):
            star = self.star_array[i]
            star_data = copy.copy(stardata[i])
            star_data[0] = math.radians(star_data[0])
            star_data[1] = math.radians(star_data[1])

            z_star = math.sin(star_data[1])
            x_star = math.cos(star_data[1]) * math.cos(star_data[0] - math.pi)
            y_star = -math.cos(star_data[1]) * math.sin(star_data[0] - math.pi)
            vec = [x_star, y_star, z_star]
            vec2 = [x_star, -y_star, z_star]
            if np.dot(vec, cam_direction) < np.dot(vec2, cam_direction):
                vec = vec2

            pixel_factor = 10
            # Always keep satellite in center to emulate large distances
            star.location = np.asarray(vec) * R + sat_pos_rel
            star.scale = (pixelsize_at_R / pixel_factor, pixelsize_at_R / pixel_factor,
                          pixelsize_at_R / pixel_factor)

            flux = math.pow(10, -0.4 * (star_data[2] - 10.))
            flux0 = math.pow(10, -0.4 * (star_data[2]))
            total_flux += flux0

            star.material_slots[0].material.node_tree.nodes.get("Emission").inputs[1].default_value = flux * pixel_factor * pixel_factor

            for scene_name in scene_names:
                scene = bpy.data.scenes[scene_name]
                if star.name not in scene.objects:
                    scene.objects.link(star)
        print("{} stars set, buffer len {}".format(i, len(self.star_array)))
        if len(self.star_array) > len(stardata):
            for scene_name in scene_names:
                scene = bpy.data.scenes[scene_name]
                for i in range(len(stardata), len(self.star_array)):
                    if self.star_array[i].name in scene.objects:
                        scene.objects.unlink(self.star_array[i])

        return total_flux

    def render_stars_directly(self, stardata, cam_direction, right_vec, up_vec, res_x, res_y, filename):
        """Render given stars."""
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
            star_data[0] = math.radians(star_data[0])
            star_data[1] = math.radians(star_data[1])

            z_star = math.sin(star_data[1])
            x_star = math.cos(star_data[1]) * math.cos(star_data[0] - math.pi)
            y_star = -math.cos(star_data[1]) * math.sin(star_data[0] - math.pi)
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
        starmap2 = starmap.copy()
        starmap2 = skimage.filters.gaussian(starmap, ss / 2., multichannel=True)
        starmap3 = np.zeros((res_y, res_x, 4), np.float32)
        for c in range(0, 4):

            starmap3[:, :, c] = skimage.transform.downscale_local_mean(starmap2[:, :, c], (ss, ss)) * (ss * ss)

        write_openexr(str(filename), starmap3)

        return (total_flux, np.sum(starmap3[:, :, 0]))


def write_openexr(filename, picture):
    """Save image in OpenEXR file format."""
    file_extension = ".exr"
    if filename[-4:] != file_extension:
        filename += file_extension

    h = len(picture)
    w = len(picture[0])
    c = len(picture[0][0])

    hdr = OpenEXR.Header(w, h)
    x = OpenEXR.OutputFile(filename, hdr)

    if c == 4:
        data_r = picture[:, :, 0].tobytes()
        data_g = picture[:, :, 1].tobytes()
        data_b = picture[:, :, 2].tobytes()
        data_a = picture[:, :, 3].tobytes()
        x.writePixels({"R": data_r, "G": data_g, "B": data_b, "A": data_a})
    elif c == 3:
        data_r = picture[:, :, 0].tobytes()
        data_g = picture[:, :, 1].tobytes()
        data_b = picture[:, :, 2].tobytes()

        x.writePixels({"R": data_r, "G": data_g, "B": data_b})
    x.close()
