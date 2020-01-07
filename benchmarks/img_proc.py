"""
Benchmarks to compare scikit-image (skimage) against OpenCV performance.
Compares image resizing and gaussian filtering.
"""

from datetime import datetime
from pathlib import Path
import sys

import cv2
import logging
import numpy
import skimage
import timeit

from sispo.sim import utils

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
    raise NotImplementedError()

def benchmark_skimage(image, kernel, sigma):
    """Run benchmark of scikit-image."""
    truncation = (kernel - 1) / 2 / sigma

def run(filepath):
    logger.debug("Starting opencv vs skimage benchmarking")
    logger.debug(f"Using image {filepath}")

    raw_img = utils.read_openexr_image(filepath)

    sigma = 5
    kernel = 5

    cmd_skimage = "sk_image = np.zeros(img.shape, dtype=np.float32)"
    cmd_skimage += "\n" + "sk_image = gaussian(img, sigma, truncate=truncation, multichannel=True)"

    cmd_cv2 = "cv_image = np.zeros(img.shape, dtype=np.float32)"
    cmd_cv2 += "\n" + "cv_image = GaussianBlur(img,(kernel,kernel),sigma,sigma)"

    iterations = 10000
    times_sk = timeit.timeit(cmd_skimage, number=iterations, setup=setup_skimage)
    times_cv2 = timeit.timeit(cmd_cv2, number=iterations, setup=setup_cv2)
    
    print(f"skimage: {times_sk / iterations} s")
    print(f"opencv: {times_cv2 / iterations} s")
    print(f"Ratio Skimage/opencv: {times_sk / times_cv2}")

    cmd_skimage += "\n" + "print('SK type: ', sk_image.dtype)"
    cmd_cv2 += "\n" + "print('CV2 type: ', cv_image.dtype)"

    exec(setup_skimage + "\n" + cmd_skimage + "\n" + "utils.write_openexr_image(str(file) + '_sk', sk_image)")
    exec(setup_cv2 + "\n" + cmd_cv2 + "\n" + "utils.write_openexr_image(str(file) + '_cv', cv_image)")


    statistics = setup_skimage + "\n" + setup_cv2 + "\n" + cmd_skimage + "\n" + cmd_cv2
    statistics += "\n" + "diff = sk_image - cv_image"
    statistics += "\n" + "print('Diff min, max: ', np.min(diff), np.max(diff))"
    statistics += "\n" + "print('Alpha min sk, cv; max, sk, cv: ', np.min(sk_image[:,:,3]), np.min(cv_image[:,:,3]), np.max(sk_image[:,:,3]), np.max(cv_image[:,:,3]))"
    exec(statistics)

if __name__ == "__main__":
    try:
        img_path = Path(sys.argv[1])
    except Exception as e:
        raise RuntimeError("Include filepath as argument")

    run(img_path)