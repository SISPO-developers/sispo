"""Trajectory simulation and object rendering module."""

import copy
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
import matplotlib.pyplot as plt
import OpenEXR
import skimage.filters
import skimage.transform
import simplejson as json
import orekit
OREKIT_VM = orekit.initVM() # pylint: disable=no-member
from orekit.pyhelpers import setup_orekit_curdir
file_dir = Path(__file__).parent.resolve()
root_dir = (file_dir / ".." / "..").resolve()
orekit_data = root_dir / "data" / "orekit-data.zip"
setup_orekit_curdir(str(orekit_data))
import org.orekit.orbits as orbits # pylint: disable=import-error
import org.orekit.utils as utils # pylint: disable=imxport-error
from org.orekit.utils import PVCoordinates # pylint: disable=import-error
from org.orekit.frames import FramesFactory # pylint: disable=import-error
from org.orekit.propagation.analytical import KeplerianPropagator # pylint: disable=import-error
from org.hipparchus.geometry.euclidean.threed import Vector3D # pylint: disable=import-error
from org.orekit.propagation.events import DateDetector # pylint: disable=import-error
from org.orekit.propagation.events.handlers import RecordAndContinue # pylint: disable=import-error
from org.orekit.propagation.events.handlers import EventHandler # pylint: disable=import-error
from org.orekit.python import PythonEventHandler, PythonOrekitFixedStepHandler # pylint: disable=import-error
from org.orekit.time import AbsoluteDate, TimeScalesFactory # pylint: disable=import-error
from mpl_toolkits.mplot3d import Axes3D

import starcat
import render as bc

import sispo.utils as ut
import sssb


class Environment():
    """Simulation environment."""

    def __init__(self, name):

        self.name = name

        self.ts = TimeScalesFactory.getTDB()
        
        self.encounter_date = AbsoluteDate(2017, 8, 19, 0, 0, 0.000, self.ts)
        self.duration = 2. * 60
        self.sim_start_date = self.encounter_date.getDate().shiftedBy(-self.duration / 2.)
        self.sim_end_date = self.encounter_date.getDate().shiftedBy(self.duration / 2.)
        
        self.frame_settings = dict()
        self.frame_settings["first"] = 0
        self.frame_settings["last"] = 10
        self.frame_settings["step_size"] = 1   

        self.ref_frame = FramesFactory.getICRF()
        self.mu_sun = utils.Constants.IAU_2015_NOMINAL_SUN_GM

        self.minimum_distance = 1E5
        self.with_terminator = True
        self.with_sunnyside = False
        self.timesampler_mode = 1
        self.slowmotion_factor = 10

        self.blender_settings = dict()
        self.blender_settings["exposure"] = 1.554
        self.blender_settings["samples"] = 48

        print(f"Start {self.frame_settings['first']} "
              f"End {self.frame_settings['last']} "
              f"Skip {self.frame_settings['step_size']}")

env = Environment("Didymos")


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

    def __init__(self, start, end, steps, mode = 1, factor = 2):
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

        print(dtout)
        DateDetector.__init__(self, dtout / 2., 1., self.times)


if __name__ == "__main__":
    pass