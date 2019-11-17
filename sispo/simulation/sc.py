"""Defining behaviour of the spacecraft (sc)."""

from pathlib import Path

from astropy import units as u
import numpy as np
import orekit
from org.orekit.orbits import KeplerianOrbit # pylint: disable=import-error
from org.orekit.frames import FramesFactory # pylint: disable=import-error
from org.orekit.propagation.analytical import KeplerianPropagator # pylint: disable=import-error
from org.orekit.time import AbsoluteDate, TimeScalesFactory # pylint: disable=import-error
from org.orekit.utils import PVCoordinates # pylint: disable=import-error
from org.hipparchus.geometry.euclidean.threed import Vector3D  # pylint: disable=import-error

from simulation.cb import CelestialBody


class Spacecraft(CelestialBody):
    """Handling properties and behaviour of the spacecraft."""

    def __init__(self, name, mu, state, trj_date):
        """Currently hard implemented for SC."""

        super().__init__(name)

        self.trj_date = trj_date

        self.trajectory = KeplerianOrbit(state, self.ref_frame, self.trj_date, mu)
        self.propagator = KeplerianPropagator(self.trajectory)

        self.payload = None

    @classmethod
    def calc_encounter_state(cls,
                             sssb_state,
                             min_dist, 
                             terminator=True,
                             sunnyside=False):
        """Calculate the state of a Spacecraft at closest distance to SSSB."""
        (sssb_pos, sssb_vel) = sssb_state

        sc_pos = cls.calc_encounter_pos(
            sssb_pos, min_dist, terminator, sunnyside)
            
        sc_vel = sssb_vel.scalarMultiply(
            (sssb_vel.getNorm() - 10000.) / sssb_vel.getNorm())

        #self.logger.info("Spacecraft relative velocity: %s", sc_vel)
        #self.logger.info("Spacecraft distance from sun: %s",
        #                 sc_pos.getNorm()/Constants.IAU_2012_ASTRONOMICAL_UNIT)

        return PVCoordinates(sc_pos, sc_vel)

    @staticmethod
    def calc_encounter_pos(sssb_pos,
                           min_dist,
                           terminator=True,
                           sunnyside=False):
        """Calculate the pos of a Spacecraft at closest distance to SSSB."""
        sssb_direction = sssb_pos.normalize()

        if terminator:
            shift = sssb_direction.scalarMultiply(-0.15)
            shift = shift.add(Vector3D(0., 0., 1.))
            shift = shift.normalize()
            shift = shift.scalarMultiply(min_dist)
            sc_pos = sssb_pos.add(shift)
        else:
            if not sunnyside:
                min_dist *= -1

            sssb_direction = sssb_direction.scalarMultiply(min_dist)
            sc_pos = sssb_pos.subtract(sssb_direction)

        return sc_pos
        

class Instrument():
    """Summarizes characteristics of an instrument."""

    def __init__(self, characteristics=None):
        
        if characteristics is None:
            self.chip_noise = 10
            self.res = (2464, 2048)
            self.pixel_l = 3.45 * u.micron
            self.pixel_a = self.pixel_l ** 2 * (1 / u.pix)
            self.chip_w = self.pixel_l * self.res[0]
            self.quantum_eff = 0.25
            self.focal_l = 230 * u.mm
            self.aperture_d = 4 * u.cm
            self.aperture_a = ((2 * u.cm) ** 2 - (1.28 * u.cm) ** 2) * np.pi/4
            self.wavelength = 550 * u.nm