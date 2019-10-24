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
        self.oMVG = openmvg.OpenMVGController(self.res_dir)
        self.oMVS = openmvs.OpenMVSController(self.res_dir)

    def create_pointcloud(self):
        """Creates point cloud from images."""
        self.oMVG.analyse_images()
        self.oMVG.compute_features()
        self.oMVG.match_features()
        self.oMVG.reconstruct_seq()

    def densify_pointcloud(self):
        """Create a dense point cloud from images and point cloud."""
        self.oMVG.export_MVS()

        self.oMVS.densify_pointcloud()

    def create_textured_model(self):
        """Creates mesh, refines it and applies texture to it."""
        self.oMVS.create_mesh()
        self.oMVS.refine_mesh()
        self.oMVS.texture_mesh()

    def create_export_pointcloud(self):
        """Creates and exports pointcloud to openMVS format.

        Includes all reconstruction steps of the openMVG tool.
        """
        self.oMVG.analyse_images()
        self.oMVG.compute_features()
        self.oMVG.match_features()
        self.oMVG.reconstruct_seq()

    def densify_mesh_texture_model(self):
        """Densifies pointcloud, creates and refines mesh and testures it.

        Includes all reconstruction steps of the openMVS tool.
        """
        self.oMVS.densify_pointcloud()
        self.oMVS.create_mesh()
        self.oMVS.refine_mesh()
        self.oMVS.texture_mesh()

    def reconstruct(self):
        """
        Applies entire reconstruction pipeline
        
        Going from images over dense point cloud to textured mesh model.
        """
        self.create_pointcloud()
        self.densify_pointcloud()
        self.create_textured_model()


if __name__ == "__main__":
    recon = Reconstructor()
    recon.reconstruct()
