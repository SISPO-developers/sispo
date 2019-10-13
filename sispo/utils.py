"""Utils module contains functions possibly used by all modules."""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import OpenEXR

def check_dir(directory):
    """Resolves directory and creates it, if it doesn't existing."""
    dir_resolved = directory.resolve()

    if not dir_resolved.exists():
        Path.mkdir(dir_resolved)

    return dir_resolved


def read_vec_string(string):
    """Converts vector string into numpy array."""
    raise NotImplementedError()


def write_vec_string(vec, prec):
    """Write data vector into string."""
    o = "["

    for (n, v) in enumerate(vec):
        o += f"{v:.{prec}f}"
        if n < len(vec) - 1:
            o += ","

    return o + "]"


def read_mat_string(string):
    """Converts matrix string into numpy array."""
    raise NotImplementedError()

def write_mat_string(mat, prec):
    """Write data matrix into string."""
    o = "["

    for (n, m) in enumerate(mat):
        o += (write_vec_string(m, prec))
        if n < len(mat) - 1:
            o += ","

    return o + "]"


def serialise(o):
    """Serialise numpy arrays to json object."""
    try:
        return np.asarray(o, dtype="float64").tolist()
    except TypeError:
        try:
            return float(o)
        except TypeError:
            return str(o)

def read_openexr_image(filename):
    """Read image in OpenEXR file format."""
    filename = str(filename)

    file_extension = ".exr"
    if filename[-len(file_extension):] != file_extension:
        filename += file_extension


    image = OpenEXR.InputFile(filename)
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
    ch = ["R","G","B","A"]

    for c in range(0,channels):
        image_channel = np.frombuffer(image.channel(ch[c]), dtype=np.float32)
        image_o[:, :, c] = image_channel.reshape(resolution[1], resolution[0])
    
    return image_o


def write_openexr_image(filename, image):
    """Save image in OpenEXR file format."""
    filename = str(filename)

    file_extension = ".exr"
    if filename[-len(file_extension):] != file_extension:
        filename += file_extension

    height = len(image)
    width = len(image[0])
    channels = len(image[0][0])

    if channels == 4:
        data_r = image[:, :, 0].tobytes()
        data_g = image[:, :, 1].tobytes()
        data_b = image[:, :, 2].tobytes()
        data_a = image[:, :, 3].tobytes()
        image_data = {"R": data_r, "G": data_g, "B": data_b, "A": data_a}
    elif channels == 3:
        data_r = image[:, :, 0].tobytes()
        data_g = image[:, :, 1].tobytes()
        data_b = image[:, :, 2].tobytes()
        image_data = {"R": data_r, "G": data_g, "B": data_b}
    else:
        raise RuntimeError("Invalid number of channels of starmap image.")
    
    hdr = OpenEXR.Header(width, height)
    file_handler = OpenEXR.OutputFile(filename, hdr)
    file_handler.writePixels(image_data)
    file_handler.close()


def create_logger(name):
    """Creates a logger with the common formatting."""
    now = datetime.now().strftime("%Y-%m-%dT%H%M%S%z")
    file_name = (now + "_" + name + ".log")
    log_dir = Path(__file__).parent.parent / "data" / "logs"
    log_dir = check_dir(log_dir)
    log_file = log_dir / file_name

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(message)s")
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)
    logger.info("\n\n#################### NEW LOG ####################\n")

    return logger
