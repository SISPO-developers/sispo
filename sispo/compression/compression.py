"""
Module for compression and decompression investigations.

This module is the main contribution of my master thesis.
"""

from pathlib import Path

import utils

class CompressionError(RuntimeError):
    """Generic error class for compression errors."""
    pass


class Compressor():
    """Main class to interface compression module."""

    def __init__(self, res_dir):
        self.res_dir = res_dir
