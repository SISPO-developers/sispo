"""Trajectory simulation and object rendering module."""

import copy
import math
import os
import sys
import time
from pathlib import Path
import logging

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
import sc
from cb import TimingEvent, TimeSampler

log_file_dir = ut.resolve_create_dir(root_dir / "data" / "logs")
log_file = log_file_dir / "sim.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter("%(asctime)s - %(name)s - %(funcName)s - %(message)s")
file_handler = logging.FileHandler(str(log_file))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)


class Environment():
    """Simulation environment."""

    def __init__(self, name):

        self.name = name

        self.ts = TimeScalesFactory.getTDB()
        
        self.encounter_date = AbsoluteDate(2017, 8, 19, 0, 0, 0.000, self.ts)
        self.duration = 2. * 60
        self.start_date = self.encounter_date.getDate().shiftedBy(-self.duration / 2.)
        self.end_date = self.encounter_date.getDate().shiftedBy(self.duration / 2.)
        
        self.frame_settings = dict()
        self.frame_settings["first"] = 0
        self.frame_settings["last"] = 2000
        self.frame_settings["step_size"] = 1   

        logger.info(f"Start {self.frame_settings['first']} "
                    f"End {self.frame_settings['last']} "
                    f"Skip {self.frame_settings['step_size']}")

        self.ref_frame = FramesFactory.getICRF()
        self.mu_sun = utils.Constants.IAU_2015_NOMINAL_SUN_GM

        self.minimum_distance = 1E5
        self.with_terminator = True
        self.with_sunnyside = False
        self.timesampler_mode = 1
        self.slowmotion_factor = 10

        self.render_settings = dict()
        self.render_settings["exposure"] = 1.554
        self.render_settings["samples"] = 48
        self.render_settings["device"] = "Auto"
        self.render_settings["tile"] = 512
        self.render_settings["x_res"] = 2464
        self.render_settings["y_res"] = 2048

        self.camera_settings = dict()
        self.camera_settings["color_depth"] = "32"
        self.camera_settings["lens"] = 230
        self.camera_settings["sensor"] = 3.45E-3 * self.render_settings["x_res"]

        logger.info(f"Rendering settings: "
                    f"Exposure: {self.render_settings['exposure']}; "
                    f"Samples: {self.render_settings['samples']}")


if __name__ == "__main__":
    env = Environment("Didymos")
    sssb = sssb.Sssb("Didymos", env.ref_frame)
    spacecraft = sc.Spacecraft("CI", env.ref_frame)

    time_sample_handler2 = TimingEvent().of_(TimeSampler)
    time_sampler2 = TimeSampler(env.start_date, env.end_date, env.frame_settings["last"], env.timesampler_mode,
                            factor=env.slowmotion_factor).withHandler(time_sample_handler2)
    sssb.propagator.addEventDetector(time_sampler2)

    logger.info("Propagating asteroid")
    sssb.propagator.propagate(env.start_date.getDate(), env.end_date.getDate())