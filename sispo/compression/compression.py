"""
Module for compression and decompression investigations.

This module is the main contribution of my master thesis.
"""

import bz2
import gzip
import lzma
from pathlib import Path
import zlib

import numpy as np

import utils

class CompressionError(RuntimeError):
    """Generic error class for compression errors."""
    pass


class Compressor():
    """Main class to interface compression module."""

    def __init__(self, res_dir):
        self.res_dir = res_dir
        self.image_dir = res_dir / "rendering"

        self.image_extension = ".exr"

        self.imgs = []

    def get_frame_ids(self):
        """Extract list of frame ids from file names of Composition images."""
        scene_name = "Comp"
        image_names = scene_name + "*" + self.image_extension
        file_names = self.image_dir.glob(image_names)

        ids = []
        for file_name in file_names:
            file_name = str(file_name.name).strip(self.image_extension)
            file_name = file_name.strip(scene_name)
            ids.append(file_name.strip("_"))

        return ids

    def load_images(self):
        """Load all composition images for compression."""
        self.img_ids = self.get_frame_ids()

        for id in self.img_ids:
            img_path = self.image_dir / ("Comp_" + id + self.image_extension)

            img = utils.read_openexr_image(img_path)
            self.imgs.append(img)

    def compress(self):
        """Compresses images."""
        compressed = []
        for img in self.imgs:
            img_cmp = lzma.compress(img, preset=9)
            compressed.append(img_cmp)

            with open(str(self.image_dir / self.img_ids[0]), "wb") as fp:
                fp.write(img_cmp)

        return compressed

    def decompress(self, compressed):
        """Decompresses images."""
        decompressed = []
        for img in compressed:
            img_cmp = lzma.decompress(img)
            img_cmp = np.frombuffer(img_cmp, dtype=np.float32)
            img_cmp = img_cmp.reshape((2048,2464,3))
            decompressed.append(img_cmp)

        return decompressed