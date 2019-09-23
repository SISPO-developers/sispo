"""Module to define common attributes of celestial bodies."""


import math
from pathlib import Path

import orekit
OREKIT_VM = orekit.initVM()  # pylint: disable=no-member
from orekit.pyhelpers import setup_orekit_curdir
file_dir = Path(__file__).parent.resolve()
root_dir = (file_dir / ".." / "..").resolve()
orekit_data = root_dir / "data" / "orekit-data.zip"
setup_orekit_curdir(str(orekit_data))
from org.orekit.propagation.events.handlers import EventHandler, RecordAndContinue  # pylint: disable=import-error
from org.orekit.python import PythonEventHandler  # pylint: disable=import-error
from org.orekit.propagation.events import DateDetector  # pylint: disable=import-error


class CelestialBody():

    def __init__(self, name, ref_frame):

        self.name = name

        self.ref_frame = ref_frame

        self.trajectory = None
        self.propagator = None

        self.event_handler = TimingEvent().of_(TimeSampler)
        self.time_sampler = None

        self.model_file = None

        self.pos = None
        self.vel = None

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


class TimingEvent(PythonEventHandler):
    """TimingEvent handler."""

    def __init__(self):
        """Initialise a TimingEvent handler."""
        PythonEventHandler.__init__(self)
        self.data = []
        self.events = 0

    def eventOccurred(self, s, detector, increasing):
        """Handle occured event."""
        self.events += 1
        if self.events % 100 == 0:
            print(f"{s.getDate()} : event {self.events}")

        self.data.append(s)
        return EventHandler.Action.CONTINUE

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
        dt = duration / (steps - 1)
        dtout = dt
        self.times = []
        t = 0.
        self.recorder = RecordAndContinue()

        if mode == 1:
            for _ in range(0, steps):
                self.times.append(start.getDate().shiftedBy(t))
                t += dt

        elif mode == 2:
            halfdur = duration / 2.

            for _ in range(0, steps):
                t2 = halfdur + math.sinh((t - halfdur) * factor / halfdur) \
                    * halfdur / math.sinh(factor)
                self.times.append(start.getDate().shiftedBy(t2))
                t += dt
            dtout = duration * math.sinh(factor / steps) / math.sinh(factor)

        DateDetector.__init__(self, dtout / 2., 1., self.times)
