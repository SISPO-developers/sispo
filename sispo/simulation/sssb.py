"""Defining behaviour of the small solar system body (SSSB)."""

import math
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
from org.orekit.propagation.analytical import KeplerianPropagator # pylint: disable=import-error
from org.orekit.time import AbsoluteDate

from simulation.cb import CelestialBody


class Sssb(CelestialBody):
    """Handling properties and behaviour of SSSB."""

    def __init__(self, name):
        """Currently hard implemented for Didymos."""
        super().__init__(name)

        a = 1.644641475071416E+00 * utils.Constants.IAU_2012_ASTRONOMICAL_UNIT
        P = 7.703805051391988E+02 * utils.Constants.JULIAN_DAY
        e = 3.838774437558215E-01
        i = math.radians(3.408231185574551E+00)
        omega = math.radians(3.192958853076784E+02)
        Omega = math.radians(7.320940216397703E+01)
        M = math.radians(1.967164895190036E+02)

        date_initial = AbsoluteDate(2017, 8, 19, 0, 0, 0.000, self.timescale)
        mu_sun = utils.Constants.IAU_2015_NOMINAL_SUN_GM

        self.trajectory = orbits.KeplerianOrbit(a, e, i, omega, Omega, M,
                                               orbits.PositionAngle.MEAN, 
                                               self.ref_frame, date_initial, mu_sun)
        self.propagator = KeplerianPropagator(self.trajectory)

        self.model_file = root_dir / "data" / "Didymos" / "didymos2.blend"