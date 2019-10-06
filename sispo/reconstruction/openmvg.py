"""Class to control openMVG behaviour."""

from pathlib import Path

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

        logger.info("openMVG executables dir %s", str(self.openMVG_dir))

        self.res_dir = res_dir