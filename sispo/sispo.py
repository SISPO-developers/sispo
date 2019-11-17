"""
The package provides a Space Imaging Simulator for Proximity Operations (SISPO)

The package creates images of a 3D object using blender. The images are render
in a flyby scenario. UCAC4 star catalogue to create the background. Afterwards
hese images are used with openMVG and openMVS to reconstruct the 3D model and
reconstruct the trajectory.
"""

from pathlib import Path

import simulation.simulation as sim
import reconstruction.reconstruction as rc
import compression.compression as comp
import utils

import matplotlib.pyplot as plt

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    res_dir = utils.check_dir(root_dir / "data" / "results" / "Didymos")
    compr = comp.Compressor(res_dir, "png")
    compr.get_frame_ids()
    compr.load_images()
    cmp = compr.compress(compr.imgs[0])
    decomp = compr.decompress(cmp)

    plt.imshow(compr.imgs[0])
    plt.show()
    plt.imshow(decomp)
    plt.show()

    print((compr.imgs[0] == decomp).all())