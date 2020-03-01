"""
The package provides a Space Imaging Simulator for Proximity Operations (SISPO)

The package creates images of a 3D object using blender. The images are render
in a flyby scenario. UCAC4 star catalogue to create the background. Afterwards
hese images are used with openMVG and openMVS to reconstruct the 3D model and
reconstruct the trajectory.
"""

import argparse
from datetime import datetime
import cProfile
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

from .sim import *
from .sim import utils
from .reconstruction import *
from .compression import *

logger = logging.getLogger("sispo")
logger.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(funcName)s - %(message)s")

parser = argparse.ArgumentParser(description=__file__.__doc__)
parser.add_argument("-i",
                    action="store",
                    default=None,
                    type=str, 
                    help="Definition file")
parser.add_argument("-v",
                    action="store_false",
                    help="Verbose output, displays log also on STDOUT")
parser.add_argument("--cli",
                    action="store_true",
                    help="If set, starts interactive cli tool")
parser.add_argument("--no-sim",
                    action="store_const",
                    const=False,
                    default=True,
                    dest="with_sim",
                    help="If set, sispo will not simulate the scenario")
parser.add_argument("--no-render",
                    action="store_const",
                    const=False,
                    default=True,
                    dest="with_render",
                    help="If set, sispo will not render the scenario")
parser.add_argument("--no-compression",
                    action="store_const",
                    const=False,
                    default=True,
                    dest="with_compression",
                    help="If set, images will not be compressed after rendering")
parser.add_argument("--no-reconstruction",
                    action="store_const",
                    const=False,
                    default=True,
                    dest="with_reconstruction",
                    help="If set, no 3D model will be reconstructed")
parser.add_argument("--sim-only",
                    action="store_true",
                    help="Will only simulate, not perform other steps.")
parser.add_argument("--sim-render-only",
                    action="store_true",
                    help="Will only simulate and render, not perform other steps.")
parser.add_argument("--render-only",
                    action="store_true",
                    help="Will only render, not perform other steps.")
parser.add_argument("--compress-only",
                    action="store_true",
                    help="Will only compress images, not perform other steps.")
parser.add_argument("--compress-reconstruct-only",
                    action="store_true",
                    help="Will only compress images and reconstruct 3D model, not perform other steps.")
parser.add_argument("--reconstruct-only",
                    action="store_true",
                    help="Will only reconstruct 3D, not perform other steps.")
parser.add_argument("--profile",
                    action="store_true",
                    help="Use cProfiler and write results to log.")

def read_input(args):
    """Read input, either from CLI or from input file"""

    if args.cli:
        settings = read_input_cli()
    else:
        settings = read_input_file(args)

    return settings


def read_input_cli():
    """
    Reads input interactively from CLI.
    """
    raise NotImplementedError()


def read_input_file(args):
    """
    Reads input from a given file.

    :type filename: String
    :param filename: Filename of a mission definition file.
    """
    if args.i is None:
        root_dir = Path(__file__).resolve().parent.parent
        def_file = root_dir / "data" / "input" / "definition.json"
    else:
        def_file = Path(args.i).resolve()

        if not def_file.exists():
            root_dir = Path(__file__).resolve().parent.parent
            def_file = root_dir / "data" / "input" / args.i.name

    with open(str(def_file), "r") as cfg_file:
        settings = json.load(cfg_file)

    return parse_settings_file(args, settings)

def parse_settings_file(args, settings):
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

    _parse_paths(settings)

    if "flags" in settings:
        parser.parse_args(args=settings["flags"], namespace=args)

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
            settings[key] = _parse_flags[settings[key]]

    return settings

def main():
    """
    Main function to run when executing file
    """
    args = parser.parse_args()

    if args.v:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logger_formatter)
        logger.addHandler(stream_handler)
    
    settings = read_input(args)

    if args.profile:
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
    logger.debug("\n\n#################### NEW SISPO LOG ####################\n")

    logger.debug("Settings:")
    logger.debug(f"{settings}")
    sim_settings = settings["simulation"]
    comp_settings = settings["compression"]
    recon_settings = settings["reconstruction"]

    if args.profile:
        logger.debug("Start Profiling")
        pr.enable()

    t_start = time.time()
    
    if args.sim_only:
        logger.debug("Only simulating, no other step")
        env = Environment(**sim_settings, ext_logger=logger)
        env.simulate()
        logger.debug("Finished simulating")
        return
    
    if args.sim_render_only:
        logger.debug("Only simulating and rendering, no other step")
        env = Environment(**sim_settings, ext_logger=logger)
        env.simulate()
        env.render()
        logger.debug("Finished simulating and rendering")
        return

    if args.render_only:
        raise NotImplementedError()
        logger.debug("Only rendering, no other step")
        env = Environment(**sim_settings, ext_logger=logger)
        env.render()
        logger.debug("Finished rendering")
        return

    if args.compress_only:
        logger.debug("Only compressing, no other step")
        comp = Compressor(**comp_settings, ext_logger=logger)
        comp.comp_decomp_series()
        logger.debug("Finished compressing")
        return

    if args.compress_reconstruct_only:
        logger.debug("Only compressing and reconstructing, no other step.")
        logger.debug("Start compressing")
        comp = Compressor(**comp_settings, ext_logger=logger)
        comp.comp_decomp_series()
        logger.debug("Finished compressing")
        logger.debug("Start reconstructing")
        recon = Reconstructor(**recon_settings, ext_logger=logger)
        recon.reconstruct()
        logger.debug("Finished reconstructing")
        return

    if args.reconstruct_only:
        logger.debug("Only reconstructing, no other step")
        recon = Reconstructor(**recon_settings, ext_logger=logger)
        recon.reconstruct()
        logger.debug("Finished reconstructing")
        return
    
    logger.debug("Run full pipeline")

    if args.with_sim or args.with_render:
        logger.debug("With either simulation or rendering")
        env = Environment(**sim_settings, ext_logger=logger)
    
        if args.with_sim:
            env.simulate()
        
        if args.with_render:
            env.render()

    if args.with_compression:
        logger.debug("With compression")
        comp = Compressor(**comp_settings, ext_logger=logger)
        comp.comp_decomp_series()

    if args.with_reconstruction:
        logger.debug("With reconstruction")
        recon = Reconstructor(**recon_settings, ext_logger=logger)
        recon.reconstruct()

    t_end = time.time()
    
    if args.profile:
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

def change_arg(arg):
    """Change an argument in the argparser namespace."""
    if not isinstance(arg, list):
        arg = [arg]
    parser.parse_args(args=arg, namespace=args)


if __name__ == "__main__":
    main()
