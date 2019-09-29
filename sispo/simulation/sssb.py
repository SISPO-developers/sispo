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
from org.orekit.attitudes import Attitude, FixedRate # pylint: disable=import-error
from org.orekit.propagation.analytical import KeplerianPropagator # pylint: disable=import-error
from org.orekit.time import AbsoluteDate
from org.hipparchus.geometry.euclidean.threed import Rotation, RotationConvention, Vector3D  # pylint: disable=import-error

from simulation.cb import CelestialBody


class Sssb(CelestialBody):
    """Handling properties and behaviour of SSSB."""

    def __init__(self, name, mu, trj_date, model_file=None):
        """Currently hard implemented for Didymos."""
        super().__init__(name, model_file=model_file)

        self.trj_date = trj_date

        a = 1.644641475071416E+00 * utils.Constants.IAU_2012_ASTRONOMICAL_UNIT
        P = 7.703805051391988E+02 * utils.Constants.JULIAN_DAY
        e = 3.838774437558215E-01
        i = math.radians(3.408231185574551E+00)
        omega = math.radians(3.192958853076784E+02)
        Omega = math.radians(7.320940216397703E+01)
        M = math.radians(1.967164895190036E+02)
        rotation_rate = 2. * math.pi / (2.2593 * 3600)

        # Define trajectory/orbit
        self.trajectory = orbits.KeplerianOrbit(a, e, i, omega, Omega, M,
                                               orbits.PositionAngle.MEAN, 
                                               self.ref_frame, self.trj_date, mu)

        # Define attitude
        self.rot_conv = RotationConvention.valueOf("VECTOR_OPERATOR")
        rotation = utils.AngularCoordinates(Rotation.IDENTITY, Vector3D(0., 0., rotation_rate))
        attitude = Attitude(self.trj_date, self.ref_frame, rotation)
        attitude_provider = FixedRate(attitude)

        # Create propagator
        self.propagator = KeplerianPropagator(self.trajectory, attitude_provider)
        