"""Defining behaviour of the small solar system body (SSSB)."""

import math
from pathlib import Path

import orekit
import org.orekit.utils as utils # pylint: disable=import-error
from org.orekit.orbits import KeplerianOrbit, PositionAngle # pylint: disable=import-error
from org.orekit.attitudes import Attitude, FixedRate # pylint: disable=import-error
from org.orekit.propagation.analytical import KeplerianPropagator # pylint: disable=import-error
from org.orekit.time import AbsoluteDate
from org.hipparchus.geometry.euclidean.threed import Rotation, RotationConvention, Vector3D  # pylint: disable=import-error

from .cb import CelestialBody


class SmallSolarSystemBody(CelestialBody):
    """Handling properties and behaviour of SSSB."""

    def __init__(self, name, mu, trj, att, model_file=None):
        """Currently hard implemented for Didymos."""
        super().__init__(name, model_file=model_file)

        date = trj["date"]
        trj_date = AbsoluteDate(int(date["year"]),
                                     int(date["month"]),
                                     int(date["day"]),
                                     int(date["hour"]),
                                     int(date["minutes"]),
                                     float(date["seconds"]),
                                     self.timescale)

        if "a" and "e" and "i" and "omega" and "Omega" and "M" not in trj:
            a = 1.644641475071416E+00 * utils.Constants.IAU_2012_ASTRONOMICAL_UNIT
            P = 7.703805051391988E+02 * utils.Constants.JULIAN_DAY
            e = 3.838774437558215E-01
            i = math.radians(3.408231185574551E+00)
            omega = math.radians(3.192958853076784E+02)
            Omega = math.radians(7.320940216397703E+01)
            M = math.radians(1.967164895190036E+02)

        if "rotation_rate" not in att:
            rotation_rate = 2. * math.pi / (2.2593 * 3600)

        # Define trajectory/orbit
        self.trajectory = KeplerianOrbit(trj["a"] * utils.Constants.IAU_2012_ASTRONOMICAL_UNIT,
                                         trj["e"],
                                         math.radians(trj["i"]),
                                         math.radians(trj["omega"]),
                                         math.radians(trj["Omega"]),
                                         math.radians(trj["M"]),
                                         PositionAngle.MEAN,
                                         self.ref_frame,
                                         trj_date, 
                                         mu)

        # Define attitude
        self.rot_conv = RotationConvention.valueOf("VECTOR_OPERATOR")
        rotation_rate = 2. * math.pi / att["rotation_rate"]
        rotation = utils.AngularCoordinates(Rotation.IDENTITY, 
                                            Vector3D(0., 0., rotation_rate))
        attitude = Attitude(trj_date, self.ref_frame, rotation)
        att_provider = FixedRate(attitude)

        # Create propagator
        self.propagator = KeplerianPropagator(self.trajectory, att_provider)
        