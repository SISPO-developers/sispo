"""Class to control openMVG behaviour."""

from pathlib import Path
import subprocess

import utils

logger = utils.logger("openmvg")

class OpenMVGControllerError(RuntimeError):
    """Generic openMVG error."""
    pass


class OpenMVGController():
    """Controls behaviour of openMVG data processing."""

    def __init__(self, res_dir):
        
        self.root_dir = Path(__file__).parent.parent.parent
        self.openMVG_dir = self.root_dir / "software" / "openMVG" / "build_openMVG"
        self.openMVG_dir = self.openMVG_dir / "Windows-AMD64-Release" / "Release"
        self.sensor_database = self.root_dir / "data" / "sensor_database" / "sensor_width_camera_database.txt"

        logger.info("openMVG executables dir %s", str(self.openMVG_dir))

        self.input_dir = self.root_dir / "data" / "ImageDataset_SceauxCastle-master" / "images"
        self.res_dir = res_dir

        self.focal = 65437

    def analyse_images(self):
        """ImageListing step of reconstruction."""
        logger.info("Start Imagelisting")

        self.matches_dir = self._resolve_create_dir(self.res_dir / "matches")
        utils.check_dir(self.matches_dir)

        exe = str(self.openMVG_dir / "openMVG_main_SfMInit_ImageListing")

        ret = subprocess.run([exe,
                              "-i", str(self.input_dir), 
                              "-o", str(self.matches_dir),
                              "-d", str(self.sensor_database),
                              "-c", "1",
                              "-f", str(self.focal), 
                              "-P", 
                              "-W", "1.0;1.0;1.0;"])
        logger.info("Image analysis returned: %s", str(ret))

    def compute_features(self):
        """Compute features in the pictures."""
        logger.info("Compute features of listed images")
 
        self.sfm_data = self.matches_dir / "sfm_data.json"

        exe = str(self.openMVG_dir / "openMVG_main_ComputeFeatures")

        ret = subprocess.run([exe,
                              "-i", str(self.sfm_data),
                              "-o", str(self.matches_dir), 
                              "-m", "SIFT",
                              "-f", "0", 
                              "-p", "ULTRA"])
        logger.info("Feature computation returned: %s", str(ret))