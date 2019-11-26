"""
The package provides a Space Imaging Simulator for Proximity Operations (SISPO)

The package creates images of a 3D object using blender. The images are render
in a flyby scenario. UCAC4 star catalogue to create the background. Afterwards
hese images are used with openMVG and openMVS to reconstruct the 3D model and
reconstruct the trajectory.
"""

import json
from pathlib import Path

import simulation.simulation as sim
import reconstruction.reconstruction as rc
import compression.compression as comp

import matplotlib.pyplot as plt

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    mission_def = root_dir / "data" / "input" / "mission_def.json"
    with open(str(mission_def), "r") as cfg_file:
        settings = json.load(cfg_file)

    env = sim.Environment(settings)
    env.simulate()
    env.render()
