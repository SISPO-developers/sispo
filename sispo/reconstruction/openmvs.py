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

    def densify_pointcloud(self,
                           p_prio=-1,
                           max_threads=0,
                           res_lvl=1,
                           res_min=640,
                           num_views=0,
                           num_views_fuse=3,
                           est_colors=False,
                           est_normals=False,
                           sample_mesh=0):
        """Increases number of points to make 3D model smoother."""
        logger.info("Densify point cloud to make model smoother")

        self.export_dir = utils.check_dir(self.res_dir / "export")
        self.export_scene = self.export_dir / "scene.mvs"
        self.export_dense = self.export_dir / "scene_densified.mvs"

        args = [str(self.openMVS_dir / "DensifyPointCloud")]
        args.extend(["-i", str(self.export_scene)])
        args.extend(["-o", str(self.export_dense)])

        args.extend(["--process-priority", str(p_prio)])
        args.extend(["--max-threads", str(max_threads)])
        args.extend(["--resolution-level", str(res_lvl)])
        args.extend(["--min-resolution", str(res_min)])
        args.extend(["--number-views", str(num_views)])
        args.extend(["--number-views-fuse", str(num_views_fuse)])
        args.extend(["--estimate-colors", str(int(est_colors))])
        args.extend(["--estimate-normals", str(int(est_normals))])
        args.extend(["--sample-mesh", str(sample_mesh)])

        ret = subprocess.run(args)
        logger.info("Point cloud densification returned: %s", str(ret))


    def create_mesh(self,
                    p_prio=-1,
                    max_threads=0,
                    const_weight=1,
                    free_space=0,
                    thickness=1,
                    quality=1,
                    decimate=1,
                    spurious=20,
                    spikes=True,
                    holes=30,
                    smooth=2):
        """Create a mesh from a 3D point cloud."""
        logger.info("Create mesh from point cloud")

        self.export_mesh = self.export_dir / "mesh.mvs"

        args = [str(self.openMVS_dir / "ReconstructMesh")]
        args.extend("-i", str(self.export_dense))
        args.extend("-o", str(self.export_mesh))

        args.extend(["--process-priority", str(p_prio)])
        args.extend(["--max-threads", str(max_threads)])
        args.extend(["--constant-weight", str(const_weight)])
        args.extend(["-f", str(free_space)])
        args.extend(["-thickness-factor", str(thickness)])
        args.extend(["--quality-factor", str(quality)])
        args.extend(["--decimate", str(decimate)])
        args.extend(["--remove-spurious", str(spurious)])
        args.extend(["--remove-spikes", str(int(spikes))])
        args.extend(["--close-holes", str(holes)])
        args.extend(["--smooth", str(smooth)])

        ret = subprocess.run(args)
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

    def texture_mesh(self):
        """Add texture to mesh using images."""
        logger.info("Add texture to mesh using images")

        self.export_scene_dense_mesh_refined = self.export_dir / "scene_dense_mesh_refined.mvs"

        exe = str(self.openMVS_dir / "TextureMesh")

        ret = subprocess.run([exe,
                              "-i", str(self.export_scene_dense_mesh),
                              "--export-type", "obj"])
        logger.info("Adding texture returned: %s", str(ret))
