"""
The compositor module combines the different output files of the simulation.

As the simulation module outputs different files for background and foreground
and because the intensity of the blender rendered images are not constant, the
compositor is required to fix the intensity issue and add the star background.
"""

import glob
import numpy as np
from pathlib import Path

import utils

logger = create_logger("compositor")


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
                 main=None, 
                 stars=None, 
                 sssb_only=None, 
                 sssb_const_dist=None, 
                 light_ref=None):
        self.main_scene = main
        self.stars = stars
        self.sssb_only = sssb_only
        self.sssb_const_dist = sssb_const_dist
        self.light_ref = light_ref

    def calc_ref_intensity(self):
        """Calculates reference intensitiy using the light reference scene."""

        (height, width, _) = self.light_ref.shape
        h_slice = (height // 2 - 35, height // 2 + 35)
        w_slice = (width // 2 - 35, width // 2 + 35)
        
        area = self.light_ref[h_slice[0]:h_slice[1], w_slice[0]:w_slice[1], 0]
        intensities = np.mean(area)
        return intensities

class ImageCompositor():
    """This class provides functions to combine the final simulation images."""

    def __init__(self, res_dir, filename=None):

        self.res_dir = res_dir
        self.image_dir = res_dir / "rendering"

        self.image_extension = ".exr"

        self.file_names = dict()
        scene_names = ["MainScene", "BackgroundStars", "SssbOnly", "SssbConstDist", "LightRef"]
        for name in scene_names:
            image_names = name + "*." + self.image_extension
            self.file_names[name] = glob.glob(str(self.image_dir / image_names))

        logger.info("Number of files: %d", len(self.file_names["MainScene"]))

    def read_complete_frame(self, scene):
        """Reads all images for a given single scene.
        
        This includes MainScene, BackgroundStars, SssbOnly, SssbConstDist,
        and LightRef.
        """
        frame = Frame()

        frame.main = utils.read_openexr_image(scene)
        frame.stars = utils.read_openexr_image(scene)
        frame.sssb_only = utils.read_openexr_image(scene)
        frame.sssb_const_dist = utils.read_openexr_image(scene)
        frame.light_ref = utils.read_openexr_image(scene)

        

if __name__ == "__main__":
    pass