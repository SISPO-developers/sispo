"""
The compositor module combines the different output files of the simulation.

As the simulation module outputs different files for background and foreground
and because the intensity of the blender rendered images are not constant, the
compositor is required to fix the intensity issue and add the star background.
"""

import glob
from pathlib import Path


class ImageCompositorError(RuntimeError):
    """This is a generic error for the compositor."""
    pass


class ImageCompositor():
    """This class provides functions to combine the final simulation images."""

    def __init__(self, res_dir):

        self.res_dir = res_dir
        self.image_dir = res_dir / "rendering"

        self.image_extension = ".exr"

        scene_names = ["MainScene", "BackgroundStars", "SssbOnly", "SssbConstDist", "LightRef"]

        self.file_names = dict()
        for name in scene_names:
            image_names = name + "*." + self.image_extension
            self.file_names[name] = glob.glob(str(self.image_dir / image_names))


if __name__ == "__main__":
    pass