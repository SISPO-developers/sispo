"""Defining behaviour of the small solar system body (SSSB)."""

import math

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
from org.orekit.time import AbsoluteDate, TimeScalesFactory # pylint: disable=import-error


class Sssb():
    """Handling properties and behaviour of SSSB."""

    def __init__(self):
        """Currently hard implemented for Didymos."""
        self.a = 1.644641475071416E+00 * utils.Constants.IAU_2012_ASTRONOMICAL_UNIT
        self.P = 7.703805051391988E+02 * utils.Constants.JULIAN_DAY
        self.e = 3.838774437558215E-01
        self.i = math.radians(3.408231185574551E+00)
        self.omega = math.radians(3.192958853076784E+02)
        self.Omega = math.radians(7.320940216397703E+01)
        self.M = math.radians(1.967164895190036E+02)

        utc = TimeScalesFactory.getTDB()
        date_initial = AbsoluteDate(2017, 8, 19, 0, 0, 0.000, utc)
        self.frame = FramesFactory.getICRF()
        mu_sun = utils.Constants.IAU_2015_NOMINAL_SUN_GM

        self.orbit = orbits.KeplerianOrbit(self.a, self.e, self.i, self.omega, self.Omega, self.M,
                                  orbits.PositionAngle.MEAN, self.frame, date_initial, mu_sun)
        self.propagator = KeplerianPropagator(self.orbit)

        self.pos_history = []

        @property
        def position(self, date):
            prop = self.propagator.propagate(date)
            return prop.getPVCoordinates(self.frame).getPosition()