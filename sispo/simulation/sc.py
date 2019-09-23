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

from sispo.simulation.cb import CelestialBody


class Spacecraft(CelestialBody):
    """Handling properties and behaviour of the spacecraft."""

    def __init__(self, name, ref_frame):
        """Currently hard implemented for SC."""

        super().__init__(name, ref_frame)

        self.frame = FramesFactory.getICRF()
        mu_sun = utils.Constants.IAU_2015_NOMINAL_SUN_GM

        self.trajectory = orbits.KeplerianOrbit(self.a, self.e, self.i, self.omega, self.Omega, self.M,
                                  orbits.PositionAngle.MEAN, self.frame, date_initial, mu_sun)
        self.propagator = KeplerianPropagator(self.trajectory)

        self.pos_history = []