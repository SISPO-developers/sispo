"""Utils module contains functions possibly used by all modules."""

import pathlib


def resolve_create_dir(directory):
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


def write_mat_string(vec, prec):
    """Write data matrix into string."""
    o = "["
    for (n, v) in enumerate(vec):
        o += (write_vec_string(v, prec))
        if n < len(vec) - 1:
            o += ","

    return o + "]"
