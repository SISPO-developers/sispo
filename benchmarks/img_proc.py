"""
Benchmarks to compare scikit-image (skimage) against OpenCV performance.
Compares image resizing and gaussian filtering.
"""

from datetime import datetime
from pathlib import Path
import time
import sys

import cv2
import logging
import numpy as np
import OpenEXR
import Imath
import skimage

logger = logging.getLogger("cv_skimage")
logger.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(funcName)s - %(message)s")

now = datetime.now().strftime("%Y-%m-%dT%H%M%S%z")
filename = "cv_skimage.log"
res_dir = Path(".").resolve()
res_dir = res_dir / now
Path.mkdir(res_dir)
log_file = res_dir / filename
file_handler = logging.FileHandler(str(log_file))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logger_formatter)
logger.addHandler(stream_handler)


def run_cv_gauss(image, sigma, kernel):
    return cv2.GaussianBlur(image, (kernel, kernel), sigma, sigma)


def run_cv_resize(image, scale):
    return cv2.resize(image, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)


def run_skimage_gauss(image, sigma, trunc):
    return skimage.filters.gaussian(image, sigma, truncate=trunc, multichannel=True)


def run_skimage_resize(image, scale):
    return skimage.transform.downscale_local_mean(image, (scale, scale, 1))


def benchmark_cv(raw_img, sigma, kernel, scale, iterations=10000):
    times_cv = []
    for _ in range(iterations):
        start = time.time()
        run_cv_gauss(raw_img, sigma, kernel)
        end = time.time()
        times_cv.append(end - start)

    time_cv_gauss = min(times_cv)

    times_cv = []
    for _ in range(iterations):
        start = time.time()
        run_cv_resize(raw_img, scale)
        end = time.time()
        times_cv.append(end - start)

    time_cv_resize = min(times_cv)

    return time_cv_gauss, time_cv_resize


def benchmark_sk(raw_img, sigma, trunc, scale, iterations=10000):
    times_sk = []
    for _ in range(iterations):
        start = time.time()
        run_skimage_gauss(raw_img, sigma, trunc)
        end = time.time()
        times_sk.append(end - start)

    time_sk_gauss = min(times_sk)

    times_sk = []
    for _ in range(iterations):
        start = time.time()
        run_skimage_resize(raw_img, scale)
        end = time.time()
        times_sk.append(end - start)

    time_sk_resize = min(times_sk)

    return time_sk_gauss, time_sk_resize


def benchmark(filepath, iterations=10000):
    """Executes benchmark."""
    sigma = 5
    kernel = 5
    trunc = (kernel - 1) / 2 / sigma
    scale = 4

    logger.debug("Starting opencv vs skimage benchmarking")
    logger.debug(f"Using image {filepath}")
    logger.debug(f"Gaussian Sigma: {sigma} and Kernel: {kernel}")
    logger.debug(f"Resizing scale: {scale}")
    logger.debug(f"Iterations: #{iterations}")

    raw_img = read_openexr_image(filepath)

    time_sk_gauss, time_sk_resize = benchmark_sk(
        raw_img, sigma, trunc, scale, iterations)
    time_cv_gauss, time_cv_resize = benchmark_cv(
        raw_img, sigma, kernel, scale, iterations)

    logger.debug(f"skimage gaussian filter timing: {time_sk_gauss} s")
    logger.debug(f"OpenCV gaussian filter timing: {time_cv_gauss} s")
    logger.debug(
        f"Gaussian filter timing ratio skimage/OpenCV: {time_sk_gauss / time_cv_gauss}")

    logger.debug(f"skimage resize timing: {time_sk_resize} s")
    logger.debug(f"OpenCV resize timing: {time_cv_resize} s")
    logger.debug(
        f"Resize timing ratio skimage/OpenCV: {time_sk_resize / time_cv_resize}")

    logger.debug("Gaussian filter statistics")

    img_skimage = run_skimage_gauss(raw_img, sigma, trunc)
    img_cv = run_cv_gauss(raw_img, sigma, kernel)

    write_openexr_image(res_dir / "skimage.exr", img_skimage)
    write_openexr_image(res_dir / "opencv.exr", img_cv)

    diff = img_skimage - img_cv
    logger.debug(f"Difference min: {np.min(diff)}; max: {np.max(diff)}")
    write_openexr_image(res_dir / "diff.exr", diff)

    equality = (img_skimage[:, :, 0:2] == img_cv[:, :, 0:2])
    logger.debug(f"Equality all: {equality.all()}; any: {equality.any()}")

    logger.debug(
        f"Image skimage min: {np.min(img_skimage)}; max: {np.max(img_skimage)}")
    logger.debug(f"Image OpenCV min: {np.min(img_cv)}; max: {np.max(img_cv)}")

    logger.debug("Resizing statistics")

    img_skimage = run_skimage_resize(raw_img, scale)
    img_cv = run_cv_resize(raw_img, scale)

    write_openexr_image(res_dir / "skimage_resized.exr", img_skimage)
    write_openexr_image(res_dir / "opencv_resized.exr", img_cv)

    diff = img_skimage - img_cv
    logger.debug(f"Difference min: {np.min(diff)}; max: {np.max(diff)}")
    write_openexr_image(res_dir / "resized_diff.exr", diff)

    equality = img_skimage[:, :, 0:2] == img_cv[:, :, 0:2]
    logger.debug(f"Equality all: {equality.all()}; any: {equality.any()}")

    logger.debug(
        f"Image skimage min: {np.min(img_skimage)}; max: {np.max(img_skimage)}")
    logger.debug(f"Image OpenCV min: {np.min(img_cv)}; max: {np.max(img_cv)}")


def read_openexr_image(filename):
    """Read image in OpenEXR file format into numpy array."""

    if not OpenEXR.isOpenExrFile(str(filename)):
        return None

    image = OpenEXR.InputFile(str(filename))

    if not image.isComplete():
        return None

    header = image.header()

    size = header["displayWindow"]
    resolution = (size.max.x - size.min.x + 1, size.max.y - size.min.y + 1)

    ch_info = header["channels"]
    if "R" in ch_info and "G" in ch_info and "B" in ch_info:
        if "A" in ch_info:
            channels = 4
        else:
            channels = 3
    else:
        return None

    image_o = np.zeros((resolution[1], resolution[0], channels), np.float32)

    ch = ["R", "G", "B", "A"]
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    for c in range(0, channels):
        image_channel = np.frombuffer(image.channel(ch[c], pt), np.float32)
        image_o[:, :, c] = image_channel.reshape(resolution[1], resolution[0])

    image.close()

    return image_o


def write_openexr_image(filename, image):
    """Save image in OpenEXR file format from numpy array."""
    height = len(image)
    width = len(image[0])
    channels = len(image[0][0])

    # Default header only has RGB channels
    hdr = OpenEXR.Header(width, height)

    if channels == 4:
        data_r = image[:, :, 0].tobytes()
        data_g = image[:, :, 1].tobytes()
        data_b = image[:, :, 2].tobytes()
        data_a = image[:, :, 3].tobytes()
        image_data = {"R": data_r, "G": data_g, "B": data_b, "A": data_a}
        # Add alpha channel to header
        alpha_channel = {"A": Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}
        hdr["channels"].update(alpha_channel)

    elif channels == 3:
        data_r = image[:, :, 0].tobytes()
        data_g = image[:, :, 1].tobytes()
        data_b = image[:, :, 2].tobytes()
        image_data = {"R": data_r, "G": data_g, "B": data_b}
    else:
        raise RuntimeError("Invalid number of channels of starmap image.")

    file_handler = OpenEXR.OutputFile(str(filename), hdr)
    file_handler.writePixels(image_data)
    file_handler.close()


if __name__ == "__main__":
    args = {}
    try:
        args["filepath"] = Path(sys.argv[1]).resolve()
    except Exception as e:
        raise RuntimeError("Include filepath as argument")

    try:
        args["iterations"] = int(sys.argv[2])
    except Exception as e:
        logger.debug("No number of iterations given")

    benchmark(**args)
