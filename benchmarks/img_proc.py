"""
Benchmarks to compare scikit-image (skimage) against OpenCV performance.
Compares image resizing and gaussian filtering.
"""

from datetime import datetime
from pathlib import Path
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
filename = (now + "_cv_skimage.log")
log_dir = Path(".").resolve()
log_file = log_dir / filename
file_handler = logging.FileHandler(str(log_file))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logger_formatter)
logger.addHandler(stream_handler)

def benchmark_cv(image, kernel, sigma):
    """Run benchmark of OpenCV."""
    result = np.zeros(image.shape, dtype=np.float32)
    result = cv2.GaussianBlur(image, (kernel, kernel), sigma, sigma)

    return result

def benchmark_skimage(image, kernel, sigma):
    """Run benchmark of scikit-image."""
    trunc = (kernel - 1) / 2 / sigma
    result = np.zeros(image.shape, dtype=np.float32)
    result = skimage.filters.gaussian(image, sigma, truncate=trunc, multichannel=True)

    return result

def run(filepath):
    logger.debug("Starting opencv vs skimage benchmarking")
    logger.debug(f"Using image {filepath}")

    raw_img = read_openexr_image(filepath)

    sigma = 5
    kernel = 5
    iterations = 10
    
    start = datetime.now()
    for _ in range(iterations):
        benchmark_skimage(raw_img, kernel, sigma)
    end = datetime.now()

    time_skimage = end - start

    start = datetime.now()
    for _ in range(iterations):
        benchmark_cv(raw_img, kernel, sigma)
    end = datetime.now()

    time_cv = end - start

    
    print(f"skimage: {time_skimage / iterations} s")
    print(f"opencv: {time_cv / iterations} s")
    print(f"Ratio Skimage/opencv: {time_skimage / time_cv}")

    cmd_skimage += "\n" + "print('SK type: ', sk_image.dtype)"
    cmd_cv2 += "\n" + "print('CV2 type: ', cv_image.dtype)"

    exec(setup_skimage + "\n" + cmd_skimage + "\n" + "utils.write_openexr_image(str(file) + '_sk', sk_image)")
    exec(setup_cv2 + "\n" + cmd_cv2 + "\n" + "utils.write_openexr_image(str(file) + '_cv', cv_image)")


    statistics = setup_skimage + "\n" + setup_cv2 + "\n" + cmd_skimage + "\n" + cmd_cv2
    statistics += "\n" + "diff = sk_image - cv_image"
    statistics += "\n" + "print('Diff min, max: ', np.min(diff), np.max(diff))"
    statistics += "\n" + "print('Alpha min sk, cv; max, sk, cv: ', np.min(sk_image[:,:,3]), np.min(cv_image[:,:,3]), np.max(sk_image[:,:,3]), np.max(cv_image[:,:,3]))"
    exec(statistics)

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
    
    image_o = np.zeros((resolution[1],resolution[0],channels), np.float32)
    
    ch = ["R", "G", "B", "A"]
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    for c in range(0, channels):
        image_channel = np.fromstring(image.channel(ch[c], pt), np.float32)
        image_o[:, :, c] = image_channel.reshape(resolution[1], resolution[0])
    
    image.close()

    return image_o

if __name__ == "__main__":
    try:
        img_path = Path(sys.argv[1]).resolve()
    except Exception as e:
        raise RuntimeError("Include filepath as argument")

    run(img_path)