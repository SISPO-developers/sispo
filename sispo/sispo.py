"""
The package provides a Space Imaging Simulator for Proximity Operations (SISPO)

The package creates images of a 3D object using blender. The images are render
in a flyby scenario. UCAC4 star catalogue to create the background. Afterwards
hese images are used with openMVG and openMVS to reconstruct the 3D model and
reconstruct the trajectory.
"""

import argparse
import json
from pathlib import Path
import time

from .sim import *
from .reconstruction import *
from .compression import *
from . import utils

parser = argparse.ArgumentParser(description=__file__.__doc__)
parser.add_argument("-i",
                    action="store",
                    default=None,
                    type=str, 
                    help="Definition file")
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
        raise RuntimeError("Sun model file does not exists!")

    lightref_file = settings["lightref"]["model"]["file"]
    lightref_file = Path(lightref_file)
    lightref_file.resolve()
    if not lightref_file.is_file():
        raise RuntimeError("Light reference model file does not exists!")

    sssb_file = settings["sssb"]["model"]["file"]
    sssb_file = Path(sssb_file)
    sssb_file.resolve()
    if not sssb_file.is_file():
        raise RuntimeError("SSSB model file does not exists!")

    return settings

def main():
    """
    Main function to run when executing file
    """
    args = parser.parse_args()
    
    if args.cli:
        settings = read_input()
    else:
        settings = read_input_file(args.i)

    t_start = time.time()
    
    if args.sim_only:
        env = Environment(settings)
        env.simulate()
        return
    
    if args.sim_render_only:
        env = Environment(settings)
        env.simulate()
        env.render()
        return

    if args.render_only:
        raise NotImplementedError()
        env = Environment(settings)
        env.render()
        return

    if args.compress_only:
        params = {"level": 7}
        comp = Compressor(Path(settings["res_dir"]).resolve(), 
                          img_ext="png",
                          algo="jpg",
                          settings=params)
        comp.load_images()
        comp.compress_series()
        return

    if args.reconstruct_only:
        recon = Reconstructor(settings)
        recon.reconstruct()
        return

    if args.with_sim or args.with_render:
        env = Environment(settings)

        if args.with_sim:
            env.simulate()
        
        if args.with_render:
            env.render()

    if args.with_compression:
        params = {"level": 7}
        comp = Compressor(Path(settings["res_dir"]).resolve(), 
                          img_ext="png",
                          algo="jpg",
                          settings=params)
        comp.load_images()
        comp.compress_series()

    if args.with_reconstruction:
        recon = Reconstructor(settings)
        recon.reconstruct()

    t_end = time.time()

    print(f"Total time: {t_end - t_start} s")

if __name__ == "__main__":
    main()
