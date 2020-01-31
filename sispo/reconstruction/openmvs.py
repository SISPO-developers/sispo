"""Class to control openMVS behaviour."""

from pathlib import Path
import subprocess

from . import utils


class OpenMVSControllerError(RuntimeError):
    """Generic openMVS error."""
    pass


class OpenMVSController():
    """Controls behaviour of openMVS data processing."""

    def __init__(self, res_dir, ext_logger, openMVS_dir=None):
        """."""
        self.logger = ext_logger

        root_dir = Path(__file__).parent.parent.parent
        if openMVS_dir is None:
            self.openMVS_dir = root_dir / "software" / "openMVS" / "build_openMVS"

            if (self.openMVS_dir / "bin" / "x64" / "Release").is_dir():
                self.openMVS_dir = self.openMVS_dir / "bin" / "x64" / "Release"
            elif (self.openMVS_dir / "install" / "bin").is_dir():
                self.openMVS_dir = self.openMVS_dir / "install" / "bin"
            else:
                raise OpenMVSControllerError("Could not find executables dir!")
        else:
            self.openMVS_dir = openMVS_dir

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
        self.logger.debug("Densify point cloud to make model smoother")

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

        try:
            utils.execute(args, self.logger, OpenMVSControllerError)
        except OpenMVSControllerError as e:
            pass

    def create_mesh(self,
                    export_type="obj",
                    p_prio=-1,
                    max_threads=0,
                    const_weight=1,
                    free_space=0,
                    thickness=1,
                    quality=1,
                    decimate=1,
                    remove_spurious=20,
                    remove_spikes=True,
                    close_holes=30,
                    smooth=2):
        """Create a mesh from a 3D point cloud."""
        self.logger.debug("Create mesh from point cloud")

        working_dir = utils.check_dir(self.res_dir / "mesh")
        self.mesh_scene = working_dir / "mesh.mvs"

        # If no dense point cloud exists, use exported scene
        args = [str(self.openMVS_dir / "ReconstructMesh")]
        if self.dense_scene.is_file():
            args.extend(["-i", str(self.dense_scene)])
        elif self.export_scene.is_file():
            self.logger.debug("Using exported scene instead of dense scene.")
            args.extend(["-i", str(self.export_scene)])
        else:
            raise OpenMVSControllerError("No pointcloud found, will not mesh")
        args.extend(["-o", str(self.mesh_scene)])
        args.extend(["-w", str(working_dir)])

        args.extend(["--export-type", str(export_type)])
        args.extend(["--process-priority", str(p_prio)])
        args.extend(["--max-threads", str(max_threads)])
        args.extend(["--constant-weight", str(const_weight)])
        args.extend(["-f", str(free_space)])
        args.extend(["--thickness-factor", str(thickness)])
        args.extend(["--quality-factor", str(quality)])
        args.extend(["--decimate", str(decimate)])
        args.extend(["--remove-spurious", str(remove_spurious)])
        args.extend(["--remove-spikes", str(int(remove_spikes))])
        args.extend(["--close-holes", str(close_holes)])
        args.extend(["--smooth", str(smooth)])
        
        utils.execute(args, self.logger, OpenMVSControllerError)
        
    def refine_mesh(self,
                    export_type="obj",
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
                    use_cuda=False):
        """
        Refine 3D mesh.
        
        Despite being used by default, CUDA is specifically disabled as default
        since it is known to cause problems. See also
        https://github.com/cdcseacave/openMVS/issues/378
        https://github.com/cdcseacave/openMVS/issues/230
        """
        self.logger.debug("Refine 3D mesh")

        working_dir = utils.check_dir(self.res_dir / "refined_mesh")
        self.refined_mesh = working_dir / "mesh_refined.mvs"

        args = [str(self.openMVS_dir / "RefineMesh")]
        args.extend(["-i", str(self.mesh_scene)])
        args.extend(["-o", str(self.refined_mesh)])
        args.extend(["-w", str(working_dir)])

        args.extend(["--export-type", str(export_type)])
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
        args.extend(["--use-cuda", str(int(use_cuda))])

        try:
            utils.execute(args, self.logger, OpenMVSControllerError)
        except OpenMVSControllerError as e:
            pass

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
        self.logger.debug("Add texture to mesh using images")

        working_dir = utils.check_dir(self.res_dir / "textured_mesh")
        self.textured_obj = working_dir / "textured_model.obj"

        # If no refined mesh exists, use regular mesh
        args = [str(self.openMVS_dir / "TextureMesh")]
        if self.refined_mesh.is_file():
            args.extend(["-i", str(self.refined_mesh)])
        elif self.mesh_scene.is_file():
            self.logger.debug("Using regular mesh instead of refined mesh.")
            args.extend(["-i", str(self.mesh_scene)])
        else:
            raise OpenMVSControllerError("No mesh found, will not texture")
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

        utils.execute(args, self.logger, OpenMVSControllerError)
