"""
The package provides a Space Imaging Simulator for Proximity Operations (SISPO)

The package creates images of a 3D object using blender. The images are render
in a flyby scenario. UCAC4 star catalogue to create the background. Afterwards
hese images are used with openMVG and openMVS to reconstruct the 3D model and
reconstruct the trajectory.
"""

import argparse
import cProfile
from datetime import datetime
import io
import json
import logging
###############################################################################
################## Hack to enable JPEG2000 format in OpenCV ###################
######## See https://github.com/opencv/opencv/issues/14058 for details ########
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "TRUE"
###############################################################################
from pathlib import Path
import pstats
import sys
import time

from .compression import *
from .reconstruction import *
from .sim import *
from .sim import utils


logger = logging.getLogger("sispo")
logger.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(funcName)s - %(message)s")


def read_input():
    """
    Reads CLI input and then parses input file.
    """
    args = read_input_cli()

    inputfile = _parse_input_filepath(args.i)

    settings = read_input_file(inputfile)

    if "flags" in settings:
        settings = _parse_input_flags(settings)

    settings = parse_input(settings)

    if args.o is not None:
        res_dir = Path(args.o).resolve()
        res_dir = utils.check_dir(res_dir)

        settings["res_dir"] = res_dir

    if args.n is not None:
        settings["name"] = args.n

    return settings


def read_input_cli():
    """
    Reads input from CLI.

    Represents top level input of a definition.json file
    """
    cli_parser = argparse.ArgumentParser(description=__file__.__doc__)
    cli_parser.add_argument("-i",
                            action="store",
                            default=None,
                            type=str,
                            help="Path to 'definition.json' file")
    cli_parser.add_argument("-o",
                            action="store",
                            default=None,
                            type=str,
                            help="Path to results directory")
    cli_parser.add_argument("-n",
                            action="store",
                            default=None,
                            type=str,
                            help="Name of simulation scenario")

    cli_args = cli_parser.parse_args()

    return cli_args


def read_input_file(filename):
    """
    Reads input from a given file.

    :type filename: String
    :param filename: Filename of a mission definition file.
    """
    with open(str(filename), "r") as def_file:
        settings = json.load(def_file)

    return settings


def parse_input(settings):
    """
    Parses settings from input file or CLI into correct data formats

    :type settings: dict
    :param settings: String based description of settings.
    """

    if "simulation" not in settings:
        logger.debug("No simulation settings provided!")

    if "compression" not in settings:
        logger.debug("No compression settings provided!")

    if "reconstruction" not in settings:
        logger.debug("No reconstruction settings provided!")

    settings = _parse_paths(settings)
    settings = _parse_flags(settings)

    return settings


def _parse_input_flags(settings):
    """
    Parse flags field in input
    """
    parser = argparse.ArgumentParser(description=__file__.__doc__)
    parser.add_argument("-v",
                        action="store_true",
                        help="Verbose output, displays log also on STDOUT")
    parser.add_argument("--with-sim",
                        action="store_true",
                        dest="with_sim",
                        help="If set, SISPO will simulate the scenario")
    parser.add_argument("--with-render",
                        action="store_true",
                        dest="with_render",
                        help="If set, SISPO will render the scenario")
    parser.add_argument("--with-compression",
                        action="store_true",
                        dest="with_compression",
                        help="If set, SISPO will compress images")
    parser.add_argument("--with-reconstruction",
                        action="store_true",
                        dest="with_reconstruction",
                        help="If set, SISPO will attempt reconstruction.")
    parser.add_argument("--profile",
                        action="store_true",
                        help="Use cProfiler and write results to log.")
    settings["flags"] = parser.parse_args(args=settings["flags"])

    # If all flags are false it is default case and all steps are done
    if not settings["flags"].with_sim and \
        not settings["flags"].with_render and \
        not settings["flags"].with_compression and \
        not settings["flags"].with_reconstruction:

        settings["flags"].with_sim = True
        settings["flags"].with_render = True
        settings["flags"].with_compression = True
        settings["flags"].with_reconstruction = True

    return settings


def _parse_paths(settings):
    """
    Recursively parses all settings with _dir suffix to a Path object.

    :type settings: dict
    :param settings: Dictionary containing settings
    """
    for key in settings.keys():
        if "dir" in key:
            if "res" in key:
                path = utils.check_dir(settings[key])
            else:
                path = utils.check_dir(settings[key], False)

            settings[key] = path
        elif "file" in key:
            file = Path(settings[key])
            file = file.resolve()
            if not file.is_file():
                raise RuntimeError(f"File {file} does not exist.")
            else:
                settings[key] = file
        elif isinstance(settings[key], dict):
            settings[key] = _parse_paths(settings[key])

    return settings


def _parse_flags(settings):
    """
    Recursively parses all settings containing with_ prefix to a bool.

    :type settings: dict
    :param settings: Dictionary containing settings
    """
    for key in settings.keys():
        if "with" in key:
            settings[key] = bool(settings[key])
        elif isinstance(settings[key], dict):
            settings[key] = _parse_flags(settings[key])

    return settings


def _parse_input_filepath(filepath):
    """
    Parse input file path either from CLI argument or default file path.
    """
    if filepath is None:
        root_dir = Path(__file__).resolve().parent.parent
        filename = root_dir / "data" / "input" / "definition.json"
    else:
        filename = Path(filepath).resolve()

        if not filename.exists():
            root_dir = Path(__file__).resolve().parent.parent
            filename = root_dir / "data" / "input" / filepath.name

    return filename


def serialize(o):
    """
    Serializes Path or Namespace objects into strings or dicts respectively.
    """
    if isinstance(o, Path):
        return str(o)
    elif isinstance(o, argparse.Namespace):
        return vars(o)
    raise TypeError(f"Object of type {type(o)} not serializable!")


def main():
    """
    Main function to run when executing file
    """
    settings = read_input()

    if settings["flags"].v:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logger_formatter)
        logger.addHandler(stream_handler)

    if settings["flags"].profile:
        pr = cProfile.Profile()

    now = datetime.now().strftime("%Y-%m-%dT%H%M%S%z")
    filename = (now + "_sispo.log")
    log_dir = settings["res_dir"]
    if not log_dir.is_dir:
        Path.mkdir(log_dir)
    log_file = log_dir / filename
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)
    logger.debug("\n\n################### NEW SISPO LOG ###################\n")

    logger.debug("Settings:")
    logger.debug(f"{json.dumps(settings, indent=4, default=serialize)}")

    sim_settings = settings["simulation"]
    comp_settings = settings["compression"]
    recon_settings = settings["reconstruction"]

    if settings["flags"].profile:
        logger.debug("Start Profiling")
        pr.enable()

    t_start = time.time()

    logger.debug("Run full pipeline")

    if settings["flags"].with_sim or settings["flags"].with_render:
        logger.debug("With either simulation or rendering")
        env = Environment(**sim_settings, ext_logger=logger)

        if settings["flags"].with_sim:
            env.simulate()

        if settings["flags"].with_render:
            env.render()

    if settings["flags"].with_compression:
        logger.debug("With compression")
        comp = Compressor(**comp_settings, ext_logger=logger)
        comp.comp_decomp_series()

    if settings["flags"].with_reconstruction:
        logger.debug("With reconstruction")
        recon = Reconstructor(**recon_settings, ext_logger=logger)
        recon.reconstruct()

    t_end = time.time()

    if settings["flags"].profile:
        pr.disable()
        logger.debug("Stop Profile")

        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()

        logger.debug("\n##################### Pstats #####################\n")
        logger.debug("\n" + s.getvalue() + "\n")
        logger.debug("\n##################################################\n")

    logger.debug(f"Total time: {t_end - t_start} s")
    logger.debug("Finished sispo main")


def run():
    """Alias for :py:func:`main` ."""
    main()


if __name__ == "__main__":
    print("SISPO is a Python package.")
    print("Either import in Python console or SISPO executable")
