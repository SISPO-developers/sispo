"""Defining behaviour of the small solar system body (SSSB)."""

import math
from pathlib import Path

import orekit
import org.orekit.utils as utils # pylint: disable=import-error
from org.orekit.orbits import KeplerianOrbit, PositionAngle # pylint: disable=import-error
from org.orekit.attitudes import Attitude, FixedRate # pylint: disable=import-error
from org.orekit.propagation.analytical import KeplerianPropagator # pylint: disable=import-error
from org.orekit.time import AbsoluteDate
from org.hipparchus.geometry.euclidean.threed import Rotation, RotationOrder, RotationConvention, Vector3D  # pylint: disable=import-error

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


        # rotation axis
        self.axis_ra = math.radians(att["RA"]) if "RA" in att else 0.
        self.axis_dec = math.radians(att["Dec"]) if "Dec" in att else math.pi/2

        # rotation offset, zero longitude right ascension at epoch
        self.rotation_zlra = math.radians(att["ZLRA"]) if "ZLRA" in att else 0.

        # rotation angular velocity [rad/s]
        if "rotation_rate" in att:
            self.rotation_rate = att["rotation_rate"] * 2.0 * math.pi / 180.0
        else:
            self.rotation_rate = 2. * math.pi / (2.2593 * 3600)     # didymain by default

        # Define initial rotation, set rotation convention
        #  - For me, FRAME_TRANSFORM order makes more sense, the rotations are applied from left to right
        #    so that the following rotations apply on previously rotated axes  +Olli
        self.rot_conv = RotationConvention.FRAME_TRANSFORM
        init_rot = Rotation(RotationOrder.ZYZ, self.rot_conv, self.axis_ra, math.pi/2-self.axis_dec, self.rotation_zlra)

        if "r" in trj:
            self.date_history = [trj_date]
            self.pos_history = [Vector3D(*trj["r"])]
            self.vel_history = [Vector3D(*(trj["v"] if "v" in trj else [0., 0., 0.]))]
            self.rot_history = [init_rot]
            return

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

        rotation = utils.AngularCoordinates(init_rot, Vector3D(0., 0., self.rotation_rate))
        attitude = Attitude(trj_date, self.ref_frame, rotation)
        att_provider = FixedRate(attitude)

        # Create propagator
        self.propagator = KeplerianPropagator(self.trajectory, att_provider)

        # Loaded coma object, currently only used with OpenGL based rendering
        self.coma = None
