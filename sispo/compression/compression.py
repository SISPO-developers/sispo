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

    def __init__(self, res_dir, algo=None, settings=None):
        self.res_dir = res_dir
        self.image_dir = res_dir / "rendering"

        self.image_extension = ".exr"

        self.imgs = []

        if algo is None:
            algo = "lzma"
        if settings is None:
            settings = {"level": 9}

        self.select_algo(algo, settings)


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

    def load_images(self, img_ids=None):
        """Load composition images using ids."""
        if img_ids is None:
            self.img_ids = self.get_frame_ids()
        else:
            self.img_ids = img_ids

        for id in self.img_ids:
            img_path = self.image_dir / ("Comp_" + id + self.image_extension)

            img = utils.read_openexr_image(img_path)
            self.imgs.append(img)

    def compress_series(self):
        """
        Compresses multiple images using :py:meth: `.compress`
        """
        compressed = []
        for img in self.imgs:
            self.compress(img)

    def compress(self, img):
        """
        Compresses images using predefined algorithm or file format.
        
        :param img: Image to be compressed.
        :returns: A the compressed images.
        """
        img_cmp = self._comp_met(img, **self._settings)
        with open(str(self.image_dir / self.img_ids[0]), "wb") as file:
            file.write(img_cmp)

        return img_cmp

    def decompress(self):
        """
        Decompresses images using predefined algorithm or file format.

        :returns: Decompressed image.
        """
        with open(str(self.image_dir / id), "rb") as file:
            img = file.read()

        img_dcmp = self._decomp_met(img)
        img_dcmp = np.frombuffer(img_dcmp, dtype=np.float32)
        img_dcmp = img_dcmp.reshape((2048,2464,3))

        return img_dcmp

    def select_algo(self, algo, settings):
        """
        Select compression and decompression algorithm or file format.

        :param algo: string to describe algorithm or file format to use for
            image compression.
        :param settings: dictionary to describe settings for the compression
            algorithm. Default is {"level": 9}, i.e. highest compression.
        """
        if algo == "bz2":
            comp = bz2.compress
            settings["compresslevel"] = settings["level"]
            settings.pop("level")
            decomp = bz2.decompress
        elif algo == "gzip":
            comp = gzip.compress
            settings["compresslevel"] = settings["level"]
            settings.pop("level")
            decomp = gzip.decompress
        elif algo == "lzma":
            comp = lzma.compress
            settings["preset"] = settings["level"]
            settings.pop("level")
            decomp = lzma.decompress
        elif algo == "zlib":
            comp = zlib.compress
            decomp = zlib.decompress
        else:
            raise CompressionError("Unknown compression algorithm.")

        self._comp_met = comp
        self._decomp_met = decomp
        self._settings = settings
