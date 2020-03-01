"""
Module for compression and decompression investigations.

This module is the main contribution of my master thesis.
"""

import bz2
from datetime import datetime
import gzip
import logging
import lzma
from pathlib import Path
import shutil
import threading
import zlib

import cv2
import numpy as np


class CompressionError(RuntimeError):
    """Generic error class for compression errors."""
    pass


class Compressor():
    """Main class to interface compression module."""

    def __init__(self, 
                 res_dir,
                 img_dir,
                 img_ext="exr",
                 algo=None,
                 settings=None,
                 ext_logger=None):

        if ext_logger is not None:
            self.logger = ext_logger
        else:
            self.logger = self._create_logger()

        self.res_dir = self._check_dir(res_dir)
        self.image_dir = self._check_dir(img_dir)
        self.raw_dir = self._check_dir(res_dir / "raw")

        self.img_extension = "." + img_ext

        self.imgs = {}
        self.xyzs = {}
        self._res = None

        if algo is None:
            algo = "lzma"
        if settings is None:
            settings = {"level": 9}

        self.select_algo(algo, settings)
        self.algo = algo

        self.logger.debug(f"Compressing with algorithm {self.algo}")
        self.logger.debug(f"Compressing with settings {self._settings}")

        self._threads = []

    def get_frame_ids(self):
        """Extract list of frame ids from file names of Inst(rument) images."""
        scene_name = "Inst"
        image_names = scene_name + "*" + self.img_extension
        file_names = self.image_dir.glob(image_names)

        ids = []
        for file_name in file_names:
            file_name = str(file_name.name).strip(self.img_extension)
            file_name = file_name.strip(scene_name)
            ids.append(file_name.strip("_"))

        self.logger.debug(f"Found {len(ids)} frame ids")

        return ids

    def load_image(self, img_id):
        """
        Load a single image into memory.

        :type img_id: str
        :param img_id: id of the image to load
        """
        self.logger.debug(f"Load image {img_id}")
        img_path = self.image_dir / ("Inst_" + img_id + self.img_extension)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        self.imgs[img_id] = img

        if self.imgs:
            id0 = list(self.imgs.keys())[0]
            if self.imgs[id0] is not None:
                self._res = self.imgs[id0].shape
            if not img.shape == self._res:
                self.logger.debug(f"Images must have size {self._res}!")
                raise CompressionError(f"Images must have size {self._res}!")

        xyz_file = ("Inst_" + img_id + self.img_extension + ".xyz")
        xyz_path = self.image_dir / xyz_file
        if xyz_path.is_file():
            self.xyzs[img_id] = xyz_path
        else:
            self.xyzs[img_id] = None

    def load_images(self, img_ids=None):
        """Load composition images using ids."""
        if img_ids is None:
            self.img_ids = self.get_frame_ids()
        else:
            self.img_ids = img_ids

        for img_id in self.img_ids:
            self.load_image(img_id)

        self.logger.debug(f"Loaded {len(self.imgs.keys())} images")       

    def unload_image(self, img_id):
        """Unload image with given img_id, keeps ID."""
        self.imgs[img_id] = None

    def unload_images(self):
        """Unloads images to free memory, keeps IDs."""
        self.imgs = {}

    def comp_decomp_series(self, max_threads=7):
        """
        Compresses and decompresses multiple images using :py:func:comp_decomp
        """
        method = self.comp_decomp
        self.logger.debug(f"{method} img series with {max_threads} threads")

        self.img_ids = self.get_frame_ids()

        for img_id in self.img_ids:

            for thr in self._threads:
                if not thr.is_alive():
                    self._threads.pop(self._threads.index(thr))

            if len(self._threads) < max_threads - 1:
                # Allow up to 2 additional threads
                thr = threading.Thread(target=method, args=(None,img_id))
                thr.start()
                self._threads.append(thr)
            else:
                # If too many, also compress in main thread to not drop a frame
                method(None,img_id)

        for thr in self._threads:
            thr.join()

        self.unload_images()

    def comp_decomp(self, img=None, img_id=None):
        """
        Default function that applies compression and decompression.

        :param img: Image to be compressed and decompressed.
        """
        compressed_img = self.compress(img, img_id)
        decompressed_img = self.decompress(compressed_img)

        if img_id is not None:
            self.logger.debug(f"Save image {img_id}")
            filename = self.res_dir / (str(img_id) + ".png")
            params = (cv2.IMWRITE_PNG_COMPRESSION, 9)
            cv2.imwrite(str(filename), decompressed_img, params)

            if self.xyzs[img_id] is not None:
                xyz_file = str(filename) + ".xyz"
                shutil.copyfile(self.xyzs[img_id], xyz_file)
                self.logger.debug(f"Save prior file {xyz_file}")

    def compress(self, img=None, img_id=None):
        """
        Compresses images using predefined algorithm or file format.
        
        :param img: Image to be compressed.
        :returns: A compressed image.
        """
        
        if img is None and img_id is not None:
            self.load_image(img_id)
            img = self.imgs[img_id]

        img_cmp = self._comp_met(img, self._settings)

        if img_id is not None:
            filename_raw = self.raw_dir / (str(img_id) + "." + self.algo)
            with open(str(self.raw_dir / filename_raw), "wb") as file:
                file.write(img_cmp)
            
            self.unload_image(img_id)

        return img_cmp

    def decompress(self, img):
        """
        Decompresses images using predefined algorithm or file format.

        :returns: Decompressed image.
        """
        img_dcmp = self._decomp_met(img)

        return img_dcmp

    def select_algo(self, algo, settings):
        """
        Select compression and decompression algorithm or file format.

        :param algo: string to describe algorithm or file format to use for
            image compression.
        :param settings: dictionary to describe settings for the compression
            algorithm. Default is {"level": 9}, i.e. highest compression.
        """
        algo = algo.lower()

        ##### Compression algorithms #####
        if algo == "bz2":
            comp = self._decorate_builtin_compress(bz2.compress)
            settings["compresslevel"] = settings["level"]
            decomp = self._decorate_builtin_decompress(bz2.decompress)
        elif algo == "gzip":
            comp = self._decorate_builtin_compress(gzip.compress)
            settings["compresslevel"] = settings["level"]
            decomp = self._decorate_builtin_decompress(gzip.decompress)
        elif algo == "lzma":
            comp = self._decorate_builtin_compress(lzma.compress)
            settings["preset"] = settings["level"]
            decomp = self._decorate_builtin_decompress(lzma.decompress)
        elif algo == "zlib":
            comp = self._decorate_builtin_compress(zlib.compress)
            decomp = self._decorate_builtin_decompress(zlib.decompress)

        ##### File formats #####
        elif algo == "jpeg" or algo == "jpg":
            comp = self._decorate_cv_compress(cv2.imencode)
            settings["ext"] = ".jpg"
            params = (cv2.IMWRITE_JPEG_QUALITY, settings["level"] * 10)

            if "progressive" in settings:
                if isinstance(settings["progressive"], bool):
                    params += (cv2.IMWRITE_JPEG_PROGRESSIVE, 
                               settings["progressive"])
                else:
                    raise CompressionError("JPEG progressive requires bool")

            if "optimize" in settings:
                if isinstance(settings["optimize"], bool):
                    params += (cv2.IMWRITE_JPEG_OPTIMIZE, settings["optimize"])
                else:
                    raise CompressionError("JPEG optimize requires bool input")

            if "rst_interval" in settings:
                if isinstance(settings["rst_interval"], int):
                    params += (cv2.IMWRITE_JPEG_RST_INTERVAL,
                               settings["rst_interval"])
                else:
                    raise CompressionError("JPEG rst_interval requires int")

            if "luma_quality" in settings:
                if isinstance(settings["luma_quality"], int):
                    params += (cv2.IMWRITE_JPEG_LUMA_QUALITY,
                               settings["luma_quality"])
                else:
                    raise CompressionError("JPEG luma_quality requires int")

            if "chroma_quality" in settings:
                if isinstance(settings["chroma_quality"], int):
                    params += (cv2.IMWRITE_JPEG_CHROMA_QUALITY,
                               settings["chroma_quality"])
                else:
                    raise CompressionError("JPEG chroma_quality requires int")

            settings["params"] = params

            decomp = self._decorate_cv_decompress(cv2.imdecode)

        elif algo == "jpeg2000" or algo == "jp2":
            comp = self._decorate_cv_compress(cv2.imencode)
            settings["ext"] = ".jp2"
            level = int(settings["level"] * 100) # Ranges from 0 to 1000
            params = (cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, level)

            settings["params"] = params

            decomp = self._decorate_cv_decompress(cv2.imdecode)

        elif algo == "png":
            comp = self._decorate_cv_compress(cv2.imencode)
            settings["ext"] = ".png"
            params = (cv2.IMWRITE_PNG_COMPRESSION, settings["level"])

            if "strategy" in settings:
                if isinstance(settings["strategy"], int):
                    params += (cv2.IMWRITE_PNG_STRATEGY, settings["strategy"])

                else:
                    raise CompressionError("PNG strategy requires int")

            if "bilevel" in settings:
                if isinstance(settings["bilevel"], bool):
                    params += (cv2.IMWRITE_PNG_BILEVEL, settings["bilevel"])

                else:
                    raise CompressionError("PNG bilevel requires bool")

            settings["params"] = params

            decomp = self._decorate_cv_decompress(cv2.imdecode)

        elif algo == "tiff":
            # According to: http://libtiff.org/support.html
            comp = self._decorate_cv_compress(cv2.imencode)
            settings["ext"] = ".tiff"
            params = ()

            if "scheme" in settings:
                # Valid values
                # 1: None
                # 2: CCITT 1D
                # 3: CCITT Group 3
                # 4: CCITT Group 4
                # 5: LZW
                # 7: JPEG
                # Also some more experimental ones exist
                if isinstance(settings["scheme"], int):
                    params += (cv2.IMWRITE_TIFF_COMPRESSION,
                               settings["scheme"])
                
                else:
                    raise CompressionError("TIFF scheme requires int")

            if "resunit" in settings:
                if isinstance(settings["resunit"], int):
                    params += (cv2.IMWRITE_TIFF_RESUNIT, settings["resunit"])

                else:
                    raise CompressionError("TIFF resunit requries int")

            if "xdpi" in settings:
                if isinstance(settings["xdpi"], int):
                    params += (cv2.IMWRITE_TIFF_XDPI, settings["xdpi"])

                else:
                    raise CompressionError("TIFF xdpi requires int")

            if "ydpi" in settings:
                if isinstance(settings["ydpi"], int):
                    params += (cv2.IMWRITE_TIFF_XDPI, settings["ydpi"])

                else:
                    raise CompressionError("TIFF ydpi requires int")

            settings["params"] = params

            decomp = self._decorate_cv_decompress(cv2.imdecode)

        elif algo == "exr":
            comp = self._decorate_cv_compress(cv2.imencode)
            settings["ext"] = ".exr"
            params = ()

            if "type" in settings:
                if isinstance(settings["type"], int):
                    params += (cv2.IMWRITE_EXR_TYPE, settings["type"])

                else:
                    raise CompressionError("EXR type requires int")

            settings["params"] = params

            decomp = self._decorate_cv_decompress(cv2.imdecode)

        else:
            raise CompressionError("Unknown compression algorithm.")

        self._comp_met = comp
        self._decomp_met = decomp
        self._settings = settings

    @staticmethod
    def _decorate_builtin_compress(func):
        def compress(img, settings):
            if img.dtype == np.float32 and np.max(img) <= 1.:
                img_temp = img * 255
                img = img_temp.astype(np.uint8)
            elif img.dtype == np.uint16:
                img_temp = img / 255
                img = img_temp.astype(np.uint8)
            elif img.dtype == np.uint8:
                pass            
            else:
                raise RuntimeError("Invalid compression input")
            img_cmp = func(img, **settings)
            return img_cmp

        return compress

    def _decorate_builtin_decompress(self, func):
        def decompress(img):
            img_dcmp = func(img)
            img_dcmp = np.frombuffer(img_dcmp, dtype=np.uint8)
            img_dcmp = img_dcmp.reshape(self._res)
            return img_dcmp

        return decompress

    @staticmethod
    def _decorate_cv_compress(func):
        def compress(img, settings):
            if img.dtype == np.float32 and np.max(img) <= 1.:
                img_temp = img * 255
                img = img_temp.astype(np.uint8)
            elif img.dtype == np.uint16:
                img_temp = img / 255
                img = img_temp.astype(np.uint8)
            elif img.dtype == np.uint8:
                pass            
            else:
                raise RuntimeError("Invalid compression input")
            #if settings["ext"] == ".jpg":
            #    img_temp = img / 255
            #    img = img_temp.astype(np.uint8)
            _, img_cmp = func(settings["ext"], img, settings["params"])
            img_cmp = np.array(img_cmp).tobytes()
            return img_cmp
        
        return compress

    def _decorate_cv_decompress(self, func):
        def decompress(img):
            img = np.frombuffer(img, dtype=np.uint8)
            img_dcmp = func(img, cv2.IMREAD_UNCHANGED)
            return img_dcmp

        return decompress

    @staticmethod
    def _create_logger():
        """
        Creates local logger in case no external logger was provided.
        """
        now = datetime.now().strftime("%Y-%m-%dT%H%M%S%z")
        filename = (now + "_compression.log")
        log_dir = Path(__file__).resolve().parent.parent.parent 
        log_dir = log_dir / "data" / "logs"
        if not log_dir.is_dir:
            Path.mkdir(log_dir)
        log_file = log_dir / filename
        logger = logging.getLogger("compression")
        logger.setLevel(logging.DEBUG)
        logger_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(funcName)s - %(message)s")
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logger_formatter)
        logger.addHandler(file_handler)
        logger.debug("\n\n############## NEW COMPRESSION LOG ##############\n")

        return logger

    def _check_dir(self, directory, create=True):
        """
        Resolves directory and creates it, if it doesn't existing.
        
        :type directory: Path or str
        :param directory: Directory to be created if not existing

        :type create: bool
        :param create: Set to false if directory should not be created and
                       instead an exception shall be raise
        """
        self.logger.debug(f"Checking if directory {directory} exists...")
        if isinstance(directory, str):
            directory = Path(directory)

        dir_resolved = directory.resolve()

        if not dir_resolved.exists():
            if create:
                self.logger.debug(f"{directory} doesn't exist. Creating it...")
                Path.mkdir(dir_resolved)
                self.logger.debug("Finished!")
            else:
                raise RuntimeError(f"Directory {directory} does not exist!")
        else:
            self.logger.debug("Exists!")

        return dir_resolved
