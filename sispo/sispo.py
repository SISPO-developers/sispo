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
parser.add_argument("-i", action="store", default=None, type=str, 
                    help="Definition file")
parser.add_argument("--cli", action="store_true",
                    help="If set, starts interactive cli tool")
parser.add_argument("--sim", action="store_false",
                    help="If set, sispo will not simulate the scenario")
parser.add_argument("--render", action="store_false",
                    help="If set, sispo will not render the scenario")
parser.add_argument("--compression", action="store_false",
                    help="If set, images will not be compressed after rendering")
parser.add_argument("--reconstruction", action="store_false",
                    help="If set, no 3D model will be reconstructed")

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

    if args.sim or args.render:
        env = Environment(settings)

        if args.sim:
            env.simulate()
        
        if args.render:
            env.render()

    if args.compression:
        params = {"level": 7}
        comp = Compressor(Path(settings["res_dir"]).resolve(), "jpg", params)
        comp.load_images()
        comp.compress_series()

    if args.reconstruction:
        recon = Reconstructor()
        recon.reconstruct()

    t_end = time.time()

    print(f"Total time: {t_end - t_start} s")

if __name__ == "__main__":
    main()
