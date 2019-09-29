"""Defining behaviour of the spacecraft (sc)."""

from pathlib import Path

import orekit
import org.orekit.utils as utils # pylint: disable=import-error
import org.orekit.orbits as orbits # pylint: disable=import-error
from org.orekit.frames import FramesFactory # pylint: disable=import-error
from org.orekit.propagation.analytical import KeplerianPropagator # pylint: disable=import-error
from org.orekit.time import AbsoluteDate, TimeScalesFactory # pylint: disable=import-error

from simulation.cb import CelestialBody


class Spacecraft(CelestialBody):
    """Handling properties and behaviour of the spacecraft."""

    def __init__(self, name, mu, state, trj_date):
        """Currently hard implemented for SC."""

        super().__init__(name)

        self.trj_date = trj_date

        self.trajectory = orbits.KeplerianOrbit(state, self.ref_frame, self.trj_date, mu)
        self.propagator = KeplerianPropagator(self.trajectory)
        