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

    def __init__(self, 
                 res_dir,
                 focal=65437,
                 intrinsics=None,
                 cam_model=1,
                 use_prior=True,
                 prior_weights=(1.0,1.0,1.0),
                 force_compute=False,
                 descriptor="SIFT",
                 d_preset="ULTRA",
                 use_upright=True,
                 num_threads=0,
                 neighbour_ratio=0.8,
                 geo_model="f",
                 num_overlaps=3,
                 pairlist_file=None,
                 method="FASTCASCADEHASHINGL2",
                 guided=False,
                 cache_size=None,
                 first_img=None,
                 second_img=None,
                 refine_options="ADJUST_ALL",
                 match_file=None,
                 p_prio=-1,
                 res_lvl=1,
                 res_min=640,
                 num_views=0,
                 num_views_fuse=3,
                 est_colors=False,
                 est_normals=False,
                 sample_mesh=0,
                 const_weight=1,
                 free_space=0,
                 thickness=1,
                 quality=1,
                 decimate=1,
                 remove_spurious=30,
                 remove_spikes=True,
                 close_holes=30,
                 smooth=2,
                 max_views=8,
                 ensure_edge_size=1,
                 max_face_area=64,
                 scales=3,
                 scale_step=0.5,
                 reduce_memory=True,
                 alt_pair=0,
                 reg_weight=0.2,
                 rig_ela_ratio=0.9,
                 grad_step=45.05,
                 vertex_ratio=0,
                 use_cuda=False,
                 export_type="obj",
                 outlier_thres=0.6,
                 cost_smooth_ratio=0.1,
                 seam_level_global=1,
                 seam_level_local=1,
                 texture_size_multiple=0,
                 patch_heuristic=3,
                 empty_color=16744231,
                 orthographic_res=0,
                 openMVG_dir=None,
                 openMVS_dir=None,
                 ext_logger=None):
        """Initialises main directory and file structure."""

        if ext_logger is not None:
            self.logger = ext_logger
        else:
            self.logger = self._create_logger()

        self.res_dir = res_dir

        if openMVG_dir is not None:
            openMVG_dir = Path(openMVG_dir).resolve()
            if not openMVG_dir.is_dir():
                openMVG_dir = None
        else:
            openMVG_dir = None
        self.oMVG = openmvg.OpenMVGController(self.res_dir,
                                              ext_logger=self.logger,
                                              openMVG_dir=openMVG_dir)

        if openMVS_dir is not None:
            openMVS_dir = Path(openMVS_dir).resolve()
            if not openMVS_dir.is_dir():
                openMVS_dir = None
        else:
            openMVS_dir = None
        self.oMVS = openmvs.OpenMVSController(self.res_dir,
                                              ext_logger=self.logger,
                                              openMVS_dir=openMVS_dir)

        self.focal = focal
        self.intrinsics = intrinsics
        self.cam_model = cam_model
        self.use_prior = use_prior
        self.prior_weights = prior_weights
        self.force_compute = force_compute
        self.descriptor = descriptor
        self.d_preset = d_preset
        self.use_upright = use_upright
        self.num_threads = num_threads
        self.neighbour_ratio = neighbour_ratio
        self.geo_model = geo_model
        self.num_overlaps = num_overlaps
        self.pairlist_file = pairlist_file
        self.method = method
        self.guided = guided
        self.cache_size = cache_size
        self.first_img = first_img
        self.second_img = second_img
        self.refine_options = refine_options
        self.match_file = match_file
        self.p_prio = p_prio
        self.res_lvl = res_lvl
        self.res_min = res_min
        self.num_views = num_views
        self.num_views_fuse = num_views_fuse
        self.est_colors = est_colors
        self.est_normals = est_normals
        self.sample_mesh = sample_mesh
        self.const_weight = const_weight
        self.free_space = free_space
        self.thickness = thickness
        self.quality = quality
        self.decimate = decimate
        self.remove_spurious = remove_spurious
        self.remove_spikes = remove_spikes
        self.close_holes = close_holes
        self.smooth = smooth
        self.max_views = max_views
        self.ensure_edge_size = ensure_edge_size
        self.max_face_area = max_face_area
        self.scales = scales
        self.scale_step = scale_step
        self.reduce_memory = reduce_memory
        self.alt_pair = alt_pair
        self.reg_weight = reg_weight
        self.rig_ela_ratio = rig_ela_ratio
        self.grad_step = grad_step
        self.vertex_ratio = vertex_ratio
        self.use_cuda = use_cuda
        self.export_type = export_type
        self.outlier_thres = outlier_thres
        self.cost_smooth_ratio = cost_smooth_ratio
        self.seam_level_global = seam_level_global
        self.seam_level_local = seam_level_local
        self.texture_size_multiple = texture_size_multiple
        self.patch_heuristic = patch_heuristic
        self.empty_color = empty_color
        self.orthographic_res = orthographic_res

    def create_pointcloud(self):
        """Creates point cloud from images."""
        self.oMVG.analyse_images(self.focal, 
                                 self.intrinsics,
                                 self.cam_model,
                                 self.use_prior,
                                 self.prior_weights)
        self.oMVG.compute_features(self.force_compute,
                                   self.descriptor,
                                   self.d_preset,
                                   self.use_upright,
                                   self.num_threads)
        self.oMVG.match_features(self.force_compute,
                                 self.neighbour_ratio,
                                 self.geo_model,
                                 self.num_overlaps,
                                 self.pairlist_file,
                                 self.method,
                                 self.guided,
                                 self.cache_size)
        self.oMVG.reconstruct_multi(self.first_img,
                                    self.second_img,
                                    self.cam_model,
                                    self.refine_options,
                                    self.use_prior,
                                    self.match_file)

    def densify_pointcloud(self):
        """Create a dense point cloud from images and point cloud."""
        self.oMVG.export_MVS(self.num_threads)

        self.oMVS.densify_pointcloud(self.p_prio,
                                     self.num_threads,
                                     self.res_lvl,
                                     self.res_min,
                                     self.num_views,
                                     self.num_views_fuse,
                                     self.est_colors,
                                     self.est_normals,
                                     self.sample_mesh)

    def create_textured_model(self):
        """Creates mesh, refines it and applies texture to it."""
        self.oMVS.create_mesh(self.export_type,
                              self.p_prio,
                              self.num_threads,
                              self.const_weight,
                              self.free_space,
                              self.thickness,
                              self.quality,
                              self.decimate,
                              self.remove_spurious,
                              self.remove_spikes,
                              self.close_holes,
                              self.smooth)
        self.oMVS.refine_mesh(self.export_type,
                              self.p_prio,
                              self.num_threads,
                              self.res_lvl,
                              self.res_min,
                              self.max_views,
                              self.decimate,
                              self.close_holes,
                              self.ensure_edge_size,
                              self.max_face_area,
                              self.scales,
                              self.scale_step,
                              self.reduce_memory,
                              self.alt_pair,
                              self.reg_weight,
                              self.rig_ela_ratio,
                              self.grad_step,
                              self.vertex_ratio,
                              self.use_cuda)
        self.oMVS.texture_mesh(self.export_type,
                               self.p_prio,
                               self.num_threads,
                               self.res_lvl,
                               self.res_min,
                               self.outlier_thres,
                               self.cost_smooth_ratio,
                               self.seam_level_global,
                               self.seam_level_local,
                               self.texture_size_multiple,
                               self.patch_heuristic,
                               self.empty_color,
                               self.orthographic_res)

    def create_export_pointcloud(self):
        """Creates and exports pointcloud to openMVS format.

        Includes all reconstruction steps of the openMVG tool.
        """
        self.oMVG.analyse_images(self.focal, 
                                 self.intrinsics,
                                 self.cam_model,
                                 self.prior,
                                 self.prior_weights)
        self.oMVG.compute_features(self.force_compute,
                                   self.descriptor,
                                   self.d_preset,
                                   self.use_upright,
                                   self.num_threads)
        self.oMVG.match_features(self.force_compute,
                                 self.neighbour_ratio,
                                 self.geo_model,
                                 self.num_overlaps,
                                 self.pairlist_file,
                                 self.method,
                                 self.guided,
                                 self.cache_size)
        self.oMVG.reconstruct_multi(self.first_img,
                                    self.second_img,
                                    self.cam_model,
                                    self.refine_options,
                                    self.use_prior,
                                    self.match_file)
        self.oMVG.export_MVS(self.num_threads)

    def densify_mesh_texture_model(self):
        """Densifies pointcloud, creates and refines mesh and testures it.

        Includes all reconstruction steps of the openMVS tool.
        """
        self.oMVS.densify_pointcloud(self.p_prio,
                                     self.num_threads,
                                     self.res_lvl,
                                     self.res_min,
                                     self.num_views,
                                     self.num_views_fuse,
                                     self.est_colors,
                                     self.est_normals,
                                     self.sample_mesh)
        self.oMVS.create_mesh(self.export_type,
                              self.p_prio,
                              self.num_threads,
                              self.const_weight,
                              self.free_space,
                              self.thickness,
                              self.quality,
                              self.decimate,
                              self.remove_spurious,
                              self.remove_spikes,
                              self.holes,
                              self.smooth)
        self.oMVS.refine_mesh(self.export_type,
                              self.p_prio,
                              self.num_threads,
                              self.res_lvl,
                              self.res_min,
                              self.max_views,
                              self.decimate,
                              self.holes,
                              self.ensure_edge_size,
                              self.max_face_area,
                              self.scales,
                              self.scale_step,
                              self.reduce_memory,
                              self.alt_pair,
                              self.reg_weight,
                              self.rig_ela_ratio,
                              self.grad_step,
                              self.vertex_ratio,
                              self.use_cuda)
        self.oMVS.texture_mesh(self.export_type,
                               self.p_prio,
                               self.num_threads,
                               self.res_lvl,
                               self.res_min,
                               self.outlier_thres,
                               self.cost_smooth_ratio,
                               self.seam_level_global,
                               self.seam_level_local,
                               self.texture_size_multiple,
                               self.patch_heuristic,
                               self.empty_color,
                               self.orthographic_res)

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
