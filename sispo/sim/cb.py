"""Module to define common attributes of celestial bodies."""

import math
import logging
from pathlib import Path

import orekit
from org.orekit.propagation.events.handlers import RecordAndContinue  # pylint: disable=import-error
from org.orekit.python import PythonEventHandler  # pylint: disable=import-error
from org.orekit.propagation.events import DateDetector  # pylint: disable=import-error
from org.orekit.frames import FramesFactory  # pylint: disable=import-error
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # pylint: disable=import-error
from org.hipparchus.ode.events import Action # pylint: disable=import-error

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(funcName)s - %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CelestialBodyError(RuntimeError):
    """Generic error for CelestialBody and child classes."""
    pass


class CelestialBody():
    """Parent class for celestial bodies such as satellites or asteroids."""

    def __init__(self, name, model_file=None):

        self.name = name

        self.timescale = TimeScalesFactory.getTDB()
        self.ref_frame = FramesFactory.getICRF()

        self.trj_date = None
        self.trajectory = None
        self.propagator = None

        self.event_handler = TimingEvent().of_(TimeSampler)
        self.time_sampler = None

        self.model_file = model_file
        self.render_obj = None

        self.pos = None
        self.vel = None

        self.date_history = self.event_handler.date_history
        self.pos_history = self.event_handler.pos_history
        self.vel_history = self.event_handler.vel_history
        self.rot_history = self.event_handler.rot_history

    def __repr__(self):
        """Objects are represented by their name."""
        return self.name

    def get_position(self, date=None):
        """Get position on given date or last calculated."""
        if date is not None:
            prop = self.propagator.propagate(date)
            self.pos = prop.getPVCoordinates(self.ref_frame).getPosition()
        return self.pos

    def get_velocity(self, date=None):
        """Get velocity on given date or last calculated."""
        if date is not None:
            prop = self.propagator.propagate(date)
            self.vel = prop.getPVCoordinates(self.ref_frame).getVelocity()
        return self.vel

    def get_state(self, date=None):
        """Get spacecraft state (position, velocity)."""
        return (self.get_position(date), self.get_velocity(date))

    def propagate(self, start, end, steps, mode=1, factor=2):
        """Propagates CB either at given start time or from start to end.

        If start and end are given start is shifted a bit earlier to detect
        event at start. end is shifted a bit later to detect event at end.
        """
        self.setup_timesampler(start, end, steps, mode, factor)

        if end is None:
            self.propagator.propagate(start)

        elif None not in (start, end):
            # TODO: check value of shift, technically 1s should be enough
            shifted_start = start.shiftedBy(-60.0)
            shifted_end = end.shiftedBy(60.0)

            self.propagator.propagate(shifted_start, shifted_end)

        else:
            raise CelestialBodyError("Invalid arguments for propagation.")

    def setup_timesampler(self, start, end, steps, mode=1, factor=2):
        """Create and attach TimeSampler to propagator."""
        self.time_sampler = TimeSampler(start, end, steps, mode, factor).withHandler(
            self.event_handler
        )
        self.propagator.addEventDetector(self.time_sampler)


class TimingEvent(PythonEventHandler):
    """TimingEvent handler."""

    def __init__(self):
        """Initialise a TimingEvent handler."""
        PythonEventHandler.__init__(self)
        self.date_history = []
        self.pos_history = []
        self.vel_history = []
        self.rot_history = []
        self.events = 0

    def eventOccurred(self, s, detector, increasing):
        """Handle occured event."""
        self.events += 1
        if self.events % 1 == 0:
            print(f"{s.getDate()} : event {self.events}")

        self.date_history.append(s.getDate())
        self.pos_history.append(s.getPVCoordinates().getPosition())
        self.vel_history.append(s.getPVCoordinates().getVelocity())
        self.rot_history.append(s.getAttitude().getRotation())
        return Action.CONTINUE

    def resetState(self, detector, oldState):
        """Reset TimingEvent handler to given state."""
        return oldState


class TimeSampler(DateDetector):
    """TimeSampler implementation."""

    def __init__(self, start, end, steps, mode=1, factor=2):
        """Initialise TimeSampler.

        mode=1 linear time, mode=2 double exponential time
        """

        duration = end.durationFrom(start)
        dtime = duration / (steps - 1)
        dtout = dtime
        self.times = []
        time = 0.0
        self.recorder = RecordAndContinue()

        if mode == 1:
            for _ in range(0, steps):
                self.times.append(start.shiftedBy(time))
                time += dtime

        elif mode == 2:
            halfdur = duration / 2.0

            for _ in range(0, steps):
                time2 = (halfdur + math.sinh(
                    (time - halfdur) * factor / halfdur
                    ) * halfdur / math.sinh(factor))
                self.times.append(start.shiftedBy(time2))
                time += dtime
            dtout = duration * math.sinh(factor / steps) / math.sinh(factor)

        DateDetector.__init__(self, dtout / 2.0, 1.0, self.times)
