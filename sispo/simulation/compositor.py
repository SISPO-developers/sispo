"""
The compositor module combines the different output files of the simulation.

As the simulation module outputs different files for background and foreground
and because the intensity of the blender rendered images are not constant, the
compositor is required to fix the intensity issue and add the star background.
"""


import numpy as np
from pathlib import Path

import utils

logger = utils.create_logger("compositor")


class ImageCompositorError(RuntimeError):
    """This is a generic error for the compositor."""
    pass


class Frame():
    """Class to wrap all data of a single frame."""

    main_scene = None
    stars = None
    sssb_only = None
    sssb_const_dist = None
    light_ref = None

    def __init__(self,
                 frame_id,
                 image_dir=None,
                 main=None,
                 stars=None,
                 sssb_only=None,
                 sssb_const_dist=None,
                 light_ref=None):

        self.id = frame_id

        if None not in (main, stars, sssb_only, sssb_const_dist, light_ref):
            self.main_scene = main
            self.stars = stars
            self.sssb_only = sssb_only
            self.sssb_const_dist = sssb_const_dist
            self.light_ref = light_ref

        elif image_dir is not None:
            self.read_complete_frame(self.id, image_dir)

        else:
            raise ImageCompositorError("Unable to create frame.")

    def calc_ref_intensity(self):
        """Calculates reference intensitiy using the light reference scene."""

        (height, width, _) = self.light_ref.shape
        h_slice = (height // 2 - 35, height // 2 + 35)
        w_slice = (width // 2 - 35, width // 2 + 35)

        area = self.light_ref[h_slice[0]:h_slice[1], w_slice[0]:w_slice[1], 0]
        intensities = np.mean(area)
        return intensities

    def calc_stars_stats(self):
        """Calculate star scene parameters."""
        star_c_max = []
        star_c_sum = []

        for i in range(3):
            star_c_max.append(np.max(self.stars[:, :, i]))
            star_c_sum.append(np.max(self.stars[:, :, i]))

        return (star_c_max, star_c_sum)

    def calc_sssb_stats(self, const_dist=False):
        """Calculate SSSB max and sum corrected with alpha channel.

        If const_dist is True, stats of const distant images are calculated.
        """
        if const_dist:
            sssb_max = np.max(
                self.sssb_const_dist[:, :, 0] * self.sssb_const_dist[:, :, 3])
            sssb_sum = np.sum(
                self.sssb_const_dist[:, :, 0] * self.sssb_const_dist[:, :, 3])
        else:
            sssb_max = np.max(
                self.sssb_only[:, :, 0] * self.sssb_only[:, :, 3])
            sssb_sum = np.sum(
                self.sssb_only[:, :, 0] * self.sssb_only[:, :, 3])

        return (sssb_max, sssb_sum)

    def read_complete_frame(self, frame_id, image_dir):
        """Reads all images for a given frame id.

        This includes MainScene, Stars, SssbOnly, SssbConstDist, and LightRef.
        """
        frame_fmt_str = image_dir / ("{}_" + frame_id + ".exr")
        frame_fmt_str = str(frame_fmt_str)

        file_name = frame_fmt_str.format("MainScene")
        self.main = utils.read_openexr_image(file_name)

        file_name = frame_fmt_str.format("Stars")
        self.stars = utils.read_openexr_image(file_name)

        file_name = frame_fmt_str.format("SssbOnly")
        self.sssb_only = utils.read_openexr_image(file_name)
        
        file_name = frame_fmt_str.format("SssbConstDist")
        self.sssb_const_dist = utils.read_openexr_image(file_name)
        
        file_name = frame_fmt_str.format("LightRef")
        self.light_ref = utils.read_openexr_image(file_name)


class ImageCompositor():
    """This class provides functions to combine the final simulation images."""

    def __init__(self, res_dir, filename=None):

        self.res_dir = res_dir
        self.image_dir = res_dir / "rendering"

        self.image_extension = ".exr"

        self.frame_ids = self.get_frame_ids()
        self.frames = []
        
        for frame_id in self.frame_ids:
            new_frame = Frame(frame_id, self.image_dir)
            self.frames.append(new_frame)

        logger.info("Number of files: %d", len(self.frames))

    def get_frame_ids(self):
        """Extract list of frame ids from file names of SssbOnly scenes."""
        scene_name = "SssbOnly"
        image_names = scene_name + "*" + self.image_extension
        file_names = self.image_dir.glob(image_names)

        ids = []
        for file_name in file_names:
            file_name = str(file_name.name).strip(self.image_extension)
            file_name = file_name.strip(scene_name)
            ids.append(file_name.strip("_"))

        return ids

    def calc_relative_intensity_curve(self):
        """Calculates the relative intensity curve for all sssb frames."""
        only_stats = []
        const_dist_stats = []

        for frame in self.frames:
            only_stats.append(frame.calc_sssb_stats())
            const_dist_stats.append(frame.calc_sssb_stats(True))

        only_stats = np.asarray(only_stats).transpose()
        const_dist_stats = np.asarray(const_dist_stats).transpose()

        rel_intensity_curve = only_stats[2] / const_dist_stats[2]

        return rel_intensity_curve


if __name__ == "__main__":
    pass
