"""Class to control openMVG behaviour."""

from pathlib import Path
import shutil
import subprocess

from . import utils


class OpenMVGControllerError(RuntimeError):
    """Generic openMVG error."""
    pass


class OpenMVGController():
    """Controls behaviour of openMVG data processing."""

    def __init__(self, res_dir, ext_logger, openMVG_dir=None):

        self.logger = ext_logger

        root_dir = Path(__file__).parent.parent.parent
        if openMVG_dir is None:
            self.openMVG_dir = root_dir / "software" / "openMVG" / "build_openMVG"

            if (self.openMVG_dir / "Windows-AMD64-Release" / "Release").is_dir():
                self.openMVG_dir = self.openMVG_dir / "Windows-AMD64-Release" / "Release"
            elif (self.openMVG_dir / "install" / "bin").is_dir():
                self.openMVG_dir = self.openMVG_dir / "install" / "bin"
            else:
                raise OpenMVGControllerError("Could not find executables dir!")
        else:
            self.openMVG_dir = openMVG_dir
        self.sensor_database = root_dir / "data" / \
            "sensor_database" / "sensor_width_camera_database.txt"

        self.logger.debug("openMVG executables dir %s", str(self.openMVG_dir))

        #self.input_dir = root_dir / "data" / "ImageDataset_SceauxCastle-master" / "images"
        self.input_dir = res_dir / "compressed"
        self.res_dir = res_dir

    def analyse_images(self,
                       focal=65437,
                       intrinsics=None,
                       cam_model=1,
                       prior=True,
                       p_weights=(1.0, 1.0, 1.0)):
        """ImageListing step of reconstruction."""
        self.logger.debug("Start Imagelisting")

        self.matches_dir = self.res_dir / "matches"
        self.matches_dir = utils.check_dir(self.matches_dir)

        args = [str(self.openMVG_dir / "openMVG_main_SfMInit_ImageListing")]
        args.extend(["-i", str(self.input_dir)])
        args.extend(["-d", str(self.sensor_database)])
        args.extend(["-o", str(self.matches_dir)])

        args.extend(["-f", str(focal)])
        if intrinsics is not None:
            args.extend(["-k", intrinsics])
        args.extend(["-c", str(cam_model)])
        if prior:
            args.extend(["-P"])
            args.extend(["-W", ";".join([str(value) for value in p_weights])])

        utils.execute(args, self.logger, OpenMVGControllerError)

    def compute_features(self,
                         force_compute=False,
                         descriptor="SIFT",
                         d_preset="ULTRA",
                         use_upright=True,
                         num_threads=0):
        """Compute features in images."""
        self.logger.debug("Compute features of listed images")

        self.sfm_data = self.matches_dir / "sfm_data.json"

        args = [str(self.openMVG_dir / "openMVG_main_ComputeFeatures")]
        args.extend(["-i", str(self.sfm_data)])
        args.extend(["-o", str(self.matches_dir)])

        args.extend(["-f", str(int(force_compute))])
        args.extend(["-m", str(descriptor)])
        args.extend(["-u", str(int(use_upright))])
        args.extend(["-p", str(d_preset)])
        args.extend(["-n", str(num_threads)])

        utils.execute(args, self.logger, OpenMVGControllerError)

    def match_features(self,
                       force_compute=False,
                       ratio=0.8,
                       geo_model="f",
                       num_overlaps=3,
                       pairlist_file=None,
                       method="FASTCASCADEHASHINGL2",
                       guided=False,
                       cache_size=None):
        """Match computed features of images."""
        self.logger.debug("Match features of images")

        args = [str(self.openMVG_dir / "openMVG_main_ComputeMatches")]
        args.extend(["-i", str(self.sfm_data)])
        args.extend(["-o", str(self.matches_dir)])

        args.extend(["-f", str(int(force_compute))])
        args.extend(["-r", str(ratio)])
        args.extend(["-g", str(geo_model)])
        args.extend(["-v", str(num_overlaps)])
        if pairlist_file is not None:
            args.extend(["-l", str(pairlist_file)])
        args.extend(["-n", str(method)])
        args.extend(["-m", str(int(guided))])
        if cache_size is not None:
            args.extend(["-c", str(cache_size)])

        utils.execute(args, self.logger, OpenMVGControllerError)

    def reconstruct_multi(self,
                          first_image=None,
                          second_image=None,
                          cam_model=3,
                          refine_options="ADJUST_ALL",
                          prior=True,
                          match_file=None):
        """Reconstructs using all algorithms provided by OpenMVG."""
        self.logger.debug("Do multi reconstruction and select best result")

        self.reconstruct = utils.check_dir(self.res_dir / "reconstruct")

        points = {}
        points["seq1"] = self.reconstruct_seq1(first_image,
                                               second_image,
                                               cam_model,
                                               refine_options,
                                               prior,
                                               match_file)

        points["seq2"] = self.reconstruct_seq2(first_image,
                                               second_image,
                                               cam_model,
                                               refine_options,
                                               prior,
                                               match_file)

        points["glob"] = self.reconstruct_global(first_image,
                                                 second_image,
                                                 refine_options,
                                                 prior,
                                                 match_file)

        best = max(points, key=points.get, default="seq1")
        self.logger.debug(f"########################################")
        self.logger.debug(f"Best reconstruction is: {best}")
        self.logger.debug(f"Number of points: {points[best]}")
        self.logger.debug(f"All results: {points}")
        self.logger.debug(f"########################################")

        if points[best] < 1:
            raise OpenMVGControllerError("Reconstruction unsuccessful!")

        dst = self.reconstruct / "sfm_data.bin"
        if best == "seq1":
            src = self.reconstruction1_dir / "sfm_data.bin"
        elif best == "seq2":
            src = self.reconstruction2_dir / "sfm_data.bin"
        elif best == "glob":
            src = self.reconstruction3_dir / "sfm_data.bin"
        self.logger.debug(f"Copying {src} to {dst}")
        shutil.copyfile(src, dst)

    def reconstruct_seq1(self,
                         first_image=None,
                         second_image=None,
                         cam_model=3,
                         refine_options="ADJUST_ALL",
                         prior=True,
                         match_file=None):
        """Reconstruct 3D models sequentially."""
        # set manually the initial pair to avoid the prompt question
        self.logger.debug("Do incremental/sequential reconstructions")

        self.reconstruction1_dir = self.reconstruct / "raw1"
        self.reconstruction1_dir = utils.check_dir(self.reconstruction1_dir)

        args = [str(self.openMVG_dir / "openMVG_main_IncrementalSfM")]
        args.extend(["-i", str(self.sfm_data)])
        args.extend(["-m", str(self.matches_dir)])
        args.extend(["-o", str(self.reconstruction1_dir)])

        if first_image is not None:
            args.extend(["-a", str(first_image)])
        if second_image is not None:
            args.extend(["-b", str(second_image)])
        args.extend(["-c", str(cam_model)])
        args.extend(["-f", str(refine_options)])
        args.extend(["-P", str(int(prior))])
        if match_file is not None:
            args.extend(["-M", str(match_file)])

        num_points = self._reconstruct(args, "#3D points: ")

        return num_points

    def reconstruct_seq2(self,
                         first_image=None,
                         second_image=None,
                         cam_model=3,
                         refine_options="ADJUST_ALL",
                         prior=True,
                         match_file=None):
        """Reconstruct 3D models sequentially."""
        # set manually the initial pair to avoid the prompt question
        self.logger.debug("Do incremental/sequential reconstructions")

        self.reconstruction2_dir = self.reconstruct / "raw2"
        self.reconstruction2_dir = utils.check_dir(self.reconstruction2_dir)

        args = [str(self.openMVG_dir / "openMVG_main_IncrementalSfM2")]
        args.extend(["-i", str(self.sfm_data)])
        args.extend(["-m", str(self.matches_dir)])
        args.extend(["-o", str(self.reconstruction2_dir)])

        if first_image is not None:
            args.extend(["-a", str(first_image)])
        if second_image is not None:
            args.extend(["-b", str(second_image)])
        args.extend(["-c", str(cam_model)])
        args.extend(["-f", str(refine_options)])
        args.extend(["-P", str(int(prior))])
        if match_file is not None:
            args.extend(["-M", str(match_file)])

        num_points = self._reconstruct(args, "#3D points: ")

        return num_points

    def reconstruct_global(self,
                           first_image=None,
                           second_image=None,
                           refine_options="ADJUST_ALL",
                           prior=True,
                           match_file=None):
        """Reconstruct 3D models globally."""
        # set manually the initial pair to avoid the prompt question
        self.logger.debug("Do global reconstructions")

        self.reconstruction3_dir = self.reconstruct / "raw3"
        self.reconstruction3_dir = utils.check_dir(self.reconstruction3_dir)

        # Global reconstruction needs matches.e.bin file
        dst = self.matches_dir / "matches.e.bin"
        if (self.matches_dir / "matches.f.bin").is_file():
            m_file = self.matches_dir / "matches.f.bin"
        elif (self.matches_dir / "matches.g.bin").is_file():
            m_file = self.matches_dir / "matches.g.bin"
        if m_file.is_file():
            shutil.copyfile(m_file, dst)

        args = [str(self.openMVG_dir / "openMVG_main_GlobalSfM")]
        args.extend(["-i", str(self.sfm_data)])
        args.extend(["-m", str(self.matches_dir)])
        args.extend(["-o", str(self.reconstruction3_dir)])

        if first_image is not None:
            args.extend(["-a", str(first_image)])
        if second_image is not None:
            args.extend(["-b", str(second_image)])
        args.extend(["-f", str(refine_options)])
        args.extend(["-P", str(int(prior))])
        if match_file is not None:
            args.extend(["-M", str(match_file)])

        num_points = self._reconstruct(args, "#3DPoints: ")

        return num_points

    def _reconstruct(self, args, search_str):
        """Common interface for multi reconstruction approach."""
        num_points = 0
        try:
            ret = utils.execute(args, self.logger, OpenMVGControllerError)
            text = ret.stdout + "\n" + ret.stderr
            idx = text.rfind(search_str)
            if idx > 0:
                sub_str = text[idx:idx+20]
                num_points = [int(s) for s in sub_str.split() if s.isdigit()]
                num_points = num_points[0]
        except OpenMVGControllerError as e:
            pass

        return num_points

    def export_MVS(self, num_threads=0):
        """Export 3D model to MVS format."""
        self.logger.debug("Exporting MVG result to MVS format")

        input_file = self.reconstruct / "sfm_data.bin"
        self.export_dir = utils.check_dir(self.res_dir / "export")
        self.export_scene = self.export_dir / "scene.mvs"
        self.undistorted_dir = utils.check_dir(self.export_dir / "undistorted")

        args = [str(self.openMVG_dir / "openMVG_main_openMVG2openMVS")]
        args.extend(["-i", str(input_file)])
        args.extend(["-o", str(self.export_scene)])
        args.extend(["-d", str(self.undistorted_dir)])

        args.extend(["-n", str(num_threads)])

        utils.execute(args, self.logger, OpenMVGControllerError)
