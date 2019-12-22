"""
Interface for handling data from a star catalogue. Retrieve data as well as
render and write images.
"""

from pathlib import Path
import subprocess
import sys

from . import utils



class StarCatalogError(RuntimeError):
    """Generic error for star catalog module."""
    pass


class StarCatalog():
    """Class to access star catalogs and render stars."""

    def __init__(self, res_dir, ext_logger, starcat_dir=None):
        """."""

        self.logger = ext_logger

        self.root_dir = Path(__file__).parent.parent.parent
        
        if starcat_dir is None:
            self.starcat_dir = self.root_dir / "data" / "UCAC4"
        else:
            starcat_dir = Path(starcat_dir)

            try:
                starcat_dir = starcat_dir.resolve()
            except OSError as e:
                raise StarCatalogError(e)

            if not starcat_dir.is_dir():
                    starcat_dir = self.models_dir / starcat_dir.name
                    starcat_dir = starcat_dir.resolve()

            if not starcat_dir.is_dir():
                raise StarCatalogError("Given star cat dir does not exist.")
        self.starcat_dir = starcat_dir

        self.res_dir = res_dir

        exe_dir = self.root_dir / "software" / "star_cats"

        if (exe_dir / "u4test").is_file() or \
                (exe_dir / "u4test.exe").is_file():
            self.exe = exe_dir / "u4test"

        elif (exe_dir / "star_cats" / "u4test").is_file() or \
                (exe_dir / "star_cats" / "u4test.exe").is_file():
            self.exe = exe_dir / "star_cats" / "u4test"

        elif (exe_dir / "build_star_cats" / "u4test").is_file() or \
                (exe_dir / "build_star_cats" / "u4test.exe").is_file():
            self.exe = exe_dir / "build_star_cats" / "u4test"
            
        else:
            raise StarCatalogError("UCAC4 interface could not be found.")

        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        # How to suppress crash notification dialog?, Jan 14,2004 -
        # Raymond Chen"s response [1]
        if sys.platform.startswith("win"):
            self.logger.debug("Windows system, surrpressing GPF dialog.")
            import ctypes
            SEM_NOGPFAULTERRORBOX = 0x0002  # From MSDN
            ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)

    def get_stardata(self, ra, dec, width, height, filename="ucac4.txt"):
        """Retrieve star data from given field of view using UCAC4 catalog."""
        res_file = self.res_dir / filename
        res_file = res_file.with_suffix(".txt")

        command = [str(self.exe),
                   str(ra),
                   str(dec),
                   str(width),
                   str(height),
                   "-h",
                   str(self.starcat_dir),
                   str(res_file)]

        for _ in range(5):
            ret = subprocess.run(command)

            if ret.returncode > 0:
                break

            self.logger.debug("Error code from star cat %d", ret.returncode)

        with open(str(res_file), "r") as rfile:
            complete_data = rfile.readlines()

        star_data = []
        for line in complete_data[1:]:
            line_data = line.split()

            ra_star = float(line_data[1])
            dec_star = float(line_data[2])
            mag_star = float(line_data[3])

            star_data.append((ra_star, dec_star, mag_star))

        self.logger.debug("Found %d stars in catalog", len(star_data))

        return star_data


# class StarCache:
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
