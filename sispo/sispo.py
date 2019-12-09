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
from pathlib import Path
import pstats
import sys
import time

from .sim import *
from .sim import utils
from .reconstruction import *
from .compression import *

now = datetime.now().strftime("%Y-%m-%dT%H%M%S%z")
filename = (now + "_sispo.log")
log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
if not log_dir.is_dir:
    Path.mkdir(log_dir)
log_file = log_dir / filename
logger = logging.getLogger("sispo")
logger.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(funcName)s - %(message)s")
file_handler = logging.FileHandler(str(log_file))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)
logger.debug("\n\n#################### NEW SISPO LOG ####################\n")

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
parser.add_argument("--reconstruct-only",
                    action="store_true",
                    help="Will only reconstruct 3D, not perform other steps.")
parser.add_argument("--profile",
                    action="store_true",
                    help="Use cProfiler and write results to log.")

pr = cProfile.Profile()

def read_input():
    """
    Reads input interactively from CLI.
    """
    raise NotImplementedError()


def read_input_file(filename=None):
    """
    Reads input from a given file.

    :type filename: String
    :param filename: Filename of a mission definition file.
    """
    if filename is None:
        root_dir = Path(__file__).resolve().parent.parent
        def_file = root_dir / "data" / "input" / "definition.json"
    else:
        def_file = Path(filename).resolve()

        if not def_file.exists():
            root_dir = Path(__file__).resolve().parent.parent
            def_file = root_dir / "data" / "input" / filename.name

    with open(str(def_file), "r") as cfg_file:
        settings = json.load(cfg_file)

    return parse_settings(settings)

def parse_settings(settings):
    """
    Parses settings from input file or CLI into correct data formats

    :type settings: dict
    :param settings: String based description of settings.
    """
    settings["res_dir"] = utils.check_dir(settings["res_dir"])
    
    settings["starcat"] = utils.check_dir(settings["starcat"], create=False)

    sun_file = settings["sun"]["model"]["file"]
    sun_file = Path(sun_file)
    sun_file.resolve()
    if not sun_file.is_file():
        logger.debug("Sun model file does not exists!")
        raise RuntimeError("Sun model file does not exists!")

    lightref_file = settings["lightref"]["model"]["file"]
    lightref_file = Path(lightref_file)
    lightref_file.resolve()
    if not lightref_file.is_file():
        logger.debug("Light reference model file does not exists!")
        raise RuntimeError("Light reference model file does not exists!")

    sssb_file = settings["sssb"]["model"]["file"]
    sssb_file = Path(sssb_file)
    sssb_file.resolve()
    if not sssb_file.is_file():
        logger.debug("SSSB model file does not exists!")
        raise RuntimeError("SSSB model file does not exists!")

    return settings

def main():
    """
    Main function to run when executing file
    """
    logger.debug("Parsing input arguments")
    args = parser.parse_args()

    if args.v:
        logger.debug("Verbose output")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logger_formatter)
        logger.addHandler(stream_handler)
    
    if args.cli:
        logger.debug("Starting interactive CLI")
        settings = read_input()
    else:
        logger.debug("Read input (definition) file")
        settings = read_input_file(args.i)

    if args.profile:
        logger.debug("Start Profiling")
        pr.enable()

    t_start = time.time()
    
    if args.sim_only:
        logger.debug("Only simulating, no other step")
        env = Environment(settings, ext_logger=logger)
        env.simulate()
        logger.debug("Finished simulating")
        return
    
    if args.sim_render_only:
        logger.debug("Only simulating and rendering, no other step")
        env = Environment(settings, ext_logger=logger)
        env.simulate()
        env.render()
        logger.debug("Finished simulating and rendering")
        return

    if args.render_only:
        raise NotImplementedError()
        logger.debug("Only rendering, no other step")
        env = Environment(settings, ext_logger=logger)
        env.render()
        logger.debug("Finished rendering")
        return

    if args.compress_only:
        logger.debug("Only compressing, no other step")
        params = {"level": 7}
        comp = Compressor(Path(settings["res_dir"]).resolve(), 
                          img_ext="png",
                          algo="jpg",
                          settings=params,
                          ext_logger=logger)
        comp.load_images()
        comp.compress_series()
        logger.debug("Finished compressing")
        return

    if args.reconstruct_only:
        logger.debug("Only reconstructing, no other step")
        recon = Reconstructor(settings, ext_logger=logger)
        recon.reconstruct()
        logger.debug("Finished reconstructing")
        return
    
    logger.debug("Run full pipeline")

    #if args.with_sim or args.with_render:
    #    logger.debug("With either simulation or rendering")
    #    env = Environment(settings, ext_logger=logger)
#
    #    if args.with_sim:
    #        env.simulate()
    #    
    #    if args.with_render:
    #        env.render()

    #if args.with_compression:
    #    logger.debug("With compression")
    #    params = {"level": 7}
    #    comp = Compressor(Path(settings["res_dir"]).resolve(), 
    #                      img_ext="exr",
    #                      algo="jpg",
    #                      settings=params,
    #                      ext_logger=logger)
    #    comp.load_images()
    #    comp.compress_series()

    if args.with_reconstruction:
        logger.debug("With reconstruction")
        recon = Reconstructor(settings, ext_logger=logger)
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
    """Alias for main()."""
    main()
    
if __name__ == "__main__":
    main()
