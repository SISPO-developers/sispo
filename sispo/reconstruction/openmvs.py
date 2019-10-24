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
        self.openMVS_dir = self.root_dir / "software" / "openMVS" / "build_openMVS"
        self.openMVS_dir = self.openMVS_dir / "bin" / "x64" / "Release"

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

        working_dir = utils.check_dir(self.res_dir / "dense")
        self.dense_scene = working_dir / "scene_densified.mvs"

        args = [str(self.openMVS_dir / "DensifyPointCloud")]
        args.extend(["-i", str(self.export_scene)])
        args.extend(["-o", str(self.dense_scene)])
        args.extend(["-w", str(working_dir)])

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

        working_dir = utils.check_dir(self.res_dir / "mesh")
        self.mesh_scene = working_dir / "mesh.mvs"

        args = [str(self.openMVS_dir / "ReconstructMesh")]
        args.extend(["-i", str(self.dense_scene)])
        args.extend(["-o", str(self.mesh_scene)])
        args.extend(["-w", str(working_dir)])

        args.extend(["--process-priority", str(p_prio)])
        args.extend(["--max-threads", str(max_threads)])
        args.extend(["--constant-weight", str(const_weight)])
        args.extend(["-f", str(free_space)])
        args.extend(["--thickness-factor", str(thickness)])
        args.extend(["--quality-factor", str(quality)])
        args.extend(["--decimate", str(decimate)])
        args.extend(["--remove-spurious", str(spurious)])
        args.extend(["--remove-spikes", str(int(spikes))])
        args.extend(["--close-holes", str(holes)])
        args.extend(["--smooth", str(smooth)])

        ret = subprocess.run(args)
        logger.info("Mesh creation returned: %s", str(ret))

    def refine_mesh(self,
                    p_prio=-1,
                    max_threads=0,
                    res_lvl=0,
                    res_min=640,
                    max_views=8,
                    decimate=1,
                    holes=30,
                    ensure_edge_size=1,
                    max_face_area=64,
                    scales=3,
                    scale_step=0.5,
                    reduce_memory=True,
                    alt_pair=0,
                    reg_weight=0.2,
                    rig_ela_r=0.9,
                    grad_step=45.05,
                    vertex_ratio=0,
                    cuda=True):
        """Refine 3D mesh."""
        logger.info("Refine 3D mesh")

        working_dir = utils.check_dir(self.res_dir / "refined_mesh")
        self.refined_mesh = working_dir / "mesh_refined.mvs"

        args =[str(self.openMVS_dir / "RefineMesh")]
        args.extend(["-i", str(self.mesh_scene)])
        args.extend(["-o", str(self.refined_mesh)])
        args.extend(["-w", str(working_dir)])

        args.extend(["--process-priority", str(p_prio)])
        args.extend(["--max-threads", str(max_threads)])
        args.extend(["--resolution-level", str(res_lvl)])
        args.extend(["--min-resolution", str(res_min)])
        args.extend(["--max-views", str(max_views)])
        args.extend(["--decimate", str(decimate)])
        args.extend(["--close-holes", str(holes)])
        args.extend(["--ensure-edge-size", str(ensure_edge_size)])
        args.extend(["--max-face-area", str(max_face_area)])
        args.extend(["--scales", str(scales)])
        args.extend(["--scale-step", str(scale_step)])
        args.extend(["--reduce-memory", str(int(reduce_memory))])
        args.extend(["--alternate-pair", str(alt_pair)])
        args.extend(["--regularity-weight", str(reg_weight)])
        args.extend(["--rigidity-elasticity-ratio", str(rig_ela_r)])
        args.extend(["--gradient-step", str(grad_step)])
        args.extend(["--planar-vertex-ratio", str(vertex_ratio)])
        args.extend(["--use-cuda", str(int(cuda))])

        ret = subprocess.run(args)
        logger.info("Mesh refinement returned: %s", str(ret))

    def texture_mesh(self,
                     export_type="obj",
                     p_prio=-1,
                     max_threads=0,
                     res_lvl=0,
                     res_min=640,
                     outlier_thres=0.6,
                     cost_smooth_r=0.1,
                     seam_level_global=1,
                     seam_level_local=1,
                     texture_size_multiple=0,
                     patch_heuristic=3,
                     empty_color=16744231,
                     orthographic_res=0):
        """Add texture to mesh using images."""
        logger.info("Add texture to mesh using images")

        working_dir = utils.check_dir(self.res_dir / "textured_mesh")
        self.textured_obj = working_dir / "textured_model.obj"

        args = [str(self.openMVS_dir / "TextureMesh")]
        args.extend(["-i", str(self.refined_mesh)])
        args.extend(["-o", str(self.textured_obj)])
        args.extend(["-w", str(working_dir)])

        args.extend(["--export-type", str(export_type)])
        args.extend(["--process-priority", str(p_prio)])
        args.extend(["--max-threads", str(max_threads)])
        args.extend(["--resolution-level", str(res_lvl)])
        args.extend(["--min-resolution", str(res_min)])
        args.extend(["--outlier-threshold", str(outlier_thres)])
        args.extend(["--cost-smoothness-ratio", str(cost_smooth_r)])
        args.extend(["--global-seam-leveling", str(seam_level_global)])
        args.extend(["--local-seam-leveling", str(seam_level_local)])
        args.extend(["--texture-size-multiple", str(texture_size_multiple)])
        args.extend(["--patch-packing-heuristic", str(patch_heuristic)])
        args.extend(["--empty-color", str(empty_color)])
        args.extend(["--orthographic-image-resolution", str(orthographic_res)])

        ret = subprocess.run(args)
        logger.info("Adding texture returned: %s", str(ret))
