"""
Reconstruction module to create 3D models from images.

Currently this module uses openMVG and openMVS.
"""

from pathlib import Path

import reconstruction.openmvg as openmvg
import reconstruction.openmvs as openmvs


class Reconstructor():
    """Reconstruction of a 3D object from images."""

    def __init__(self):
        """Initialises main directory and file structure."""
        self.root_dir = Path(__file__).parent.parent.parent
        self.res_dir = self.root_dir / "data" / "results" / "Didymos"
        oMVG = openmvg.OpenMVGController(self.res_dir)
        oMVS = openmvs.OpenMVSController(self.res_dir)

        oMVG.analyse_images()
        oMVG.compute_features()
        oMVG.match_features()
        oMVG.reconstruct_seq()
        oMVG.export_MVS()

        oMVS.densify_pointcloud()
        oMVS.create_mesh()
        oMVS.refine_mesh()
        oMVS.texture_mesh()


if __name__ == "__main__":
    recon = Reconstructor()
