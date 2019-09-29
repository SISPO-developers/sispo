"""Utils module contains functions possibly used by all modules."""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np


def check_dir(directory):
    """Resolves directory and creates it, if it doesn't existing."""
    dir_resolved = directory.resolve()

    if not dir_resolved.exists():
        Path.mkdir(dir_resolved)

    return dir_resolved


def write_vec_string(vec, prec):
    """Write data vector into string."""
    o = "["

    for (n, v) in enumerate(vec):
        o += f"{v:.{prec}f}"
        if n < len(vec) - 1:
            o += ","

    return o + "]"


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
