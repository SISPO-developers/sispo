"""Class to control openMVS behaviour."""

from pathlib import Path
import subprocess

import utils

logger = utils.create_logger("openmvs")


class OpenMVSControllerError(RuntimeError):
    """Generic openMVS error."""
    pass


class OpenMVSController():
    """Controls behaviour of openMVS data processing."""

    def __init__(self, res_dir):
        """."""
        self.root_dir = Path(__file__).parent.parent.parent
        self.openMVS_dir = self.root_dir / "software" / "openMVS" / "build"
        self.openMVS_dir = self.openMVS_dir / "bin" / "x64" / "Debug"

        self.res_dir = res_dir

    def densify_pointcloud(self):
        """Increases number of points to make 3D model smoother."""
        logger.info("Densify point cloud to make model smoother")

        self.export_dir = utils.check_dir(self.res_dir / "export")
        self.export_scene = self.export_dir / "scene.mvs"

        exe = str(self.openMVS_dir / "DensifyPointCloud")

        ret = subprocess.run([exe,
                              "-i", str(self.export_scene),
                              "--max-threads", "0",
                              "--estimate-normals", "1",
                              "--number-views", "0",
                              "-v", "3"])#,"--number-views-fuse","5"] )
        logger.info("Point cloud densification returned: %s", str(ret))


    def create_mesh(self):
        """Create a mesh from a 3D point cloud."""
        logger.info("Create mesh from point cloud")

        self.export_scene_dense = self.export_dir / "scene_dense.mvs"

        exe = str(self.openMVS_dir / "ReconstructMesh")

        ret = subprocess.run([exe,
                              "-i", str(self.export_scene_dense)])
        logger.info("Mesh creation returned: %s", str(ret))

    def refine_mesh(self):
        """Refine 3D mesh."""
        logger.info("Refine 3D mesh")

        self.export_scene_dense_mesh = self.export_dir / "scene_dense_mesh.mvs"

        exe = str(self.openMVS_dir / "RefineMesh")

        ret = subprocess.run([exe,
                             "-i", str(self.export_scene_dense_mesh),
                             "--use-cuda", "0"])
        logger.info("Mesh refinement returned: %s", str(ret))
