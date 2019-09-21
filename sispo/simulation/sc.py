"""Defining behaviour of the spacecraft (sc)."""

from pathlib import Path

import orekit
OREKIT_VM = orekit.initVM() # pylint: disable=no-member
from orekit.pyhelpers import setup_orekit_curdir
file_dir = Path(__file__)
root_dir = file_dir / ".." / ".." / ".."
orekit_data = root_dir / "data" / "orekit-data.zip"
setup_orekit_curdir(str(orekit_data))
import org.orekit.utils as utils # pylint: disable=import-error
import org.orekit.orbits as orbits # pylint: disable=import-error
from org.orekit.frames import FramesFactory # pylint: disable=import-error
from org.orekit.propagation.analytical import KeplerianPropagator # pylint: disable=import-error
from org.orekit.time import AbsoluteDate, TimeScalesFactory # pylint: disable=import-error

class Spacecraft():
    """Handling properties and behaviour of the spacecraft."""

    def __init__(self, ROOT_DIR_PATH):
        """Currently hard implemented for SC."""

        self.frame = FramesFactory.getICRF()
        mu_sun = utils.Constants.IAU_2015_NOMINAL_SUN_GM

        self.orbit = orbits.KeplerianOrbit(self.a, self.e, self.i, self.omega, self.Omega, self.M,
                                  orbits.PositionAngle.MEAN, self.frame, date_initial, mu_sun)
        self.propagator = KeplerianPropagator(self.orbit)

        self.pos_history = []

        self.pos = None
        self.vel = None