"""Utils module contains functions possibly used by all modules."""

import pathlib

def resolve_create_dir(directory):
    """Resolves directory and creates it, if it doesn't existing."""
    dir_resolved = directory.resolve()

    if not dir_resolved.exists():
        Path.mkdir(dir_resolved)

    return dir_resolved