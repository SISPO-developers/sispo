"""
Reconstruction module to create 3D models from images.

Currently this module uses openMVG and openMVS.
"""

from datetime import datetime
import logging
from pathlib import Path

from . import openmvg
from . import openmvs


class Reconstructor():
    """Reconstruction of a 3D object from images."""

    def __init__(self, settings, ext_logger=None):
        """Initialises main directory and file structure."""

        if ext_logger is not None:
            self.logger = ext_logger
        else:
            self.logger = self._create_logger()

        self.res_dir = settings["res_dir"]

        if "openMVG_dir" in settings:
            openMVG_dir = Path(settings["openMVG_dir"]).resolve()
            if not openMVG_dir.is_dir():
                openMVG_dir = None
        else:
            openMVG_dir = None
        self.oMVG = openmvg.OpenMVGController(self.res_dir,
                                              ext_logger=self.logger,
                                              openMVG_dir=openMVG_dir)

        if "openMVS_dir" in settings:
            openMVS_dir = Path(settings["openMVS_dir"]).resolve()
            if not openMVS_dir.is_dir():
                openMVS_dir = None
        else:
            openMVS_dir = None
        self.oMVS = openmvs.OpenMVSController(self.res_dir,
                                              ext_logger=self.logger,
                                              openMVS_dir=openMVS_dir)

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

    @staticmethod
    def _create_logger():
        """
        Creates local logger in case no external logger was provided.
        """
        now = datetime.now().strftime("%Y-%m-%dT%H%M%S%z")
        filename = (now + "_reconstruction.log")
        log_dir = Path(__file__).resolve().parent.parent.parent 
        log_dir = log_dir / "data" / "logs"
        if not log_dir.is_dir:
            Path.mkdir(log_dir)
        log_file = log_dir / filename
        logger = logging.getLogger("reconstruction")
        logger.setLevel(logging.DEBUG)
        logger_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(funcName)s - %(message)s")
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logger_formatter)
        logger.addHandler(file_handler)
        logger.debug("\n\n############ NEW RECONSTRUCTION LOG ############\n")

        return logger


if __name__ == "__main__":
    pass
