"""Trajectory simulation and object rendering module."""

from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import orekit
from orekit.pyhelpers import setup_orekit_curdir
#################### orekit VM init ####################
FILE_DIR = Path(__file__).parent.resolve()
ROOT_DIR = FILE_DIR.parent.parent
OREKIT_DATA_FILE = ROOT_DIR / "data" / "orekit-data.zip"
OREKIT_VM = orekit.initVM() # pylint: disable=no-member
setup_orekit_curdir(str(OREKIT_DATA_FILE))
#################### orekit VM init ####################
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # pylint: disable=import-error
from org.orekit.frames import FramesFactory  # pylint: disable=import-error
from org.orekit.utils import Constants  # pylint: disable=import-error

from simulation.cb import CelestialBody
from simulation.sc import Spacecraft
from simulation.sssb import SmallSolarSystemBody
import simulation.render as render
import simulation.starcat as starcat
import utils

class Environment():
    """Simulation environment."""

    def __init__(self, name, duration):

        self.name = name
        
        self.root_dir = Path(__file__).parent.parent.parent
        self.models_dir = utils.check_dir(self.root_dir / "data" / "models")

        self.res_dir = utils.check_dir(self.root_dir / "data" / "results" / name)

        self.sta = starcat.StarCatalog(self.res_dir)

        self.logger = utils.create_logger("simulation")

        self.ts = TimeScalesFactory.getTDB()
        self.encounter_date = AbsoluteDate(2017, 8, 15, 12, 0, 0.000, self.ts)
        self.duration = duration
        self.start_date = self.encounter_date.shiftedBy(-self.duration / 2.)
        self.end_date = self.encounter_date.shiftedBy(self.duration / 2.)

        self.ref_frame = FramesFactory.getICRF()
        self.mu_sun = Constants.IAU_2015_NOMINAL_SUN_GM

        self.frame_settings = dict()
        self.frame_settings["first"] = 0
        self.frame_settings["last"] = 10
        self.frame_settings["step_size"] = 1

        self.logger.info("First frame: %d last frame: %d Step size: %d",
                    self.frame_settings['first'], self.frame_settings['last'],
                    self.frame_settings['step_size'])

        self.minimum_distance = 1E5
        self.with_terminator = False
        self.with_sunnyside = True
        self.timesampler_mode = 1
        self.slowmotion_factor = 10

        self.with_backgroundstars = True
        self.with_sssbonly = False
        self.with_sssbconstdist = False
        self.with_lightingref = False

        self.asteroid_scenes = []

        self.render_settings = dict()
        self.render_settings["exposure"] = 1.554
        self.render_settings["samples"] = 48
        self.render_settings["device"] = "GPU"
        self.render_settings["tile"] = 512
        self.render_settings["res"] = (2464, 2048)
        self.render_settings["color_depth"] = "32"

        self.camera_settings = dict()
        self.camera_settings["lens"] = 230
        self.camera_settings["sensor"] = 3.45E-3 * \
            self.render_settings["res"][0]

        self.logger.info("Rendering settings: Exposure: %d; Samples: %d",
                    self.render_settings['exposure'], self.render_settings['samples'])

        # Setup rendering engine (renderer)
        self.setup_renderer()

        # Setup Sun
        self.setup_sun()

        # Setup SSSB
        self.setup_sssb()

        # Setup SC
        self.setup_spacecraft()

        # Setup Lightref
        if self.with_lightingref:
            self.setup_lightref()

    def setup_renderer(self):
        """Create renderer, apply common settings and create sc cam."""

        render_dir = utils.check_dir(self.res_dir / "rendering")

        self.renderer = render.BlenderController(render_dir)
        self.asteroid_scenes.append("MainScene")

        if self.with_backgroundstars:
            self.renderer.create_scene("BackgroundStars")

        if self.with_sssbonly:
            self.renderer.create_scene("SssbOnly")
            self.asteroid_scenes.append("SssbOnly")

        self.renderer.create_camera("ScCam")
        self.renderer.configure_camera("ScCam", **self.camera_settings)

        if self.with_sssbconstdist:
            self.renderer.create_scene("SssbConstDist")
            self.renderer.create_camera("SssbConstDistCam", scenes="SssbConstDist")
            self.renderer.configure_camera("SssbConstDistCam", **self.camera_settings)
            self.asteroid_scenes.append("SssbConstDist")

        if self.with_lightingref:
            self.renderer.create_scene("LightRef")
            self.renderer.create_camera("LightRefCam", scenes="LightRef")
            self.renderer.configure_camera("LightRefCam", **self.camera_settings)

        self.renderer.set_device(self.render_settings["device"])
        self.renderer.set_samples(self.render_settings["samples"])
        self.renderer.set_exposure(self.render_settings["exposure"])
        self.renderer.set_resolution(self.render_settings["res"])
        self.renderer.set_output_format()

    def setup_sun(self):
        """Create Sun and respective render object."""
        sun_model_file = self.models_dir / "didymos_lowpoly.blend"
        self.sun = CelestialBody("Sun", model_file=sun_model_file)
        self.sun.render_obj = self.renderer.load_object(self.sun.model_file, self.sun.name)

    def setup_sssb(self):
        """Create SmallSolarSystemBody and respective blender object."""
        sssb_model_file = self.models_dir / "didymos2.blend"
        self.sssb = SmallSolarSystemBody("Didymos", self.mu_sun, AbsoluteDate(
            2017, 8, 19, 0, 0, 0.000, self.ts), model_file=sssb_model_file)
        self.sssb.render_obj = self.renderer.load_object(self.sssb.model_file, "Didymos.001", self.asteroid_scenes)
        self.sssb.render_obj.rotation_mode = "AXIS_ANGLE"

    def setup_spacecraft(self):
        """Create Spacecraft and respective blender object."""
        sssb_state = self.sssb.get_state(self.encounter_date)
        sc_state = Spacecraft.calc_encounter_state(sssb_state,
                                                   self.minimum_distance,
                                                   self.with_terminator,
                                                   self.with_sunnyside)
        self.spacecraft = Spacecraft(
            "CI", self.mu_sun, sc_state, self.encounter_date)

    def setup_lightref(self):
        """Create lightreference blender object."""
        lightref_model_file = self.models_dir / "didymos_lowpoly.blend"
        self.lightref = self.renderer.load_object(lightref_model_file, "CalibrationDisk", scenes="LightRef")
        self.lightref.location = (0, 0, 0)

    def simulate(self):
        """Do simulation."""
        self.logger.info("Starting simulation")

        self.sssb.setup_timesampler(
            self.start_date, self.end_date, self.frame_settings["last"],
            self.timesampler_mode, self.slowmotion_factor)
        self.spacecraft.setup_timesampler(
            self.start_date, self.end_date, self.frame_settings["last"],
            self.timesampler_mode, self.slowmotion_factor)

        self.logger.info("Propagating SSSB")
        self.sssb.propagate(self.start_date, self.end_date)

        self.logger.info("Propagating Spacecraft")
        self.spacecraft.propagate(self.start_date, self.end_date)

        self.logger.info("Simulation completed")

        self.save_results()

    def render(self):
        """Render simulation scenario."""
        self.logger.info("Rendering simulation")

        for (date, sc_pos, sssb_pos, sssb_rot) in zip(self.spacecraft.date_history,
                                                      self.spacecraft.pos_history,
                                                      self.sssb.pos_history,
                                                      self.sssb.rot_history):

            date_str = datetime.strptime(date.toString(), "%Y-%m-%dT%H:%M:%S.%f")
            date_str = date_str.strftime("%Y-%m-%dT%H%M%S-%f")

            pos_sc_rel_sssb = np.asarray(sc_pos.subtract(sssb_pos).toArray()) / 1000.
            self.renderer.set_camera_location("ScCam", pos_sc_rel_sssb)

            if self.with_sssbconstdist:
                pos_cam_const_dist = pos_sc_rel_sssb * 1000. / np.sqrt(np.dot(pos_sc_rel_sssb, pos_sc_rel_sssb))
                self.renderer.set_camera_location("SssbConstDistCam", pos_cam_const_dist)

            sssb_axis = sssb_rot.getAxis(self.sssb.rot_conv)
            sssb_angle = sssb_rot.getAngle()
            self.sssb.render_obj.rotation_axis_angle = (sssb_angle, sssb_axis.x, sssb_axis.y, sssb_axis.z)

            self.sun.render_obj.location = -np.asarray(sssb_pos.toArray()) / 1000.

            self.renderer.target_camera(self.sssb.render_obj, "ScCam")
            
            if self.with_sssbconstdist:
                self.renderer.target_camera(self.sssb.render_obj, "SssbConstDistCam")

            if self.with_lightingref:
                self.renderer.set_camera_location("LightRefCam" ,-np.asarray(sssb_pos.toArray()) * 1000. /np.sqrt(np.dot(np.asarray(sssb_pos.toArray()),np.asarray(sssb_pos.toArray()))))
                self.renderer.target_camera(self.sun.render_obj, "CalibrationDisk")
                self.renderer.target_camera(self.lightref, "LightRefCam")
            
            self.renderer.render(date_str)

            if self.with_backgroundstars:
                fov_vecs = render.get_fov_vecs("ScCam", "MainScene")
                ra, dec, width, height = render.get_fov(fov_vecs[1], fov_vecs[2], fov_vecs[3], fov_vecs[4])
                starlist = self.sta.get_stardata(ra, dec, width, height)
                fluxes = render.render_starmap(starlist, fov_vecs, self.render_settings["res"], self.res_dir / (date_str + "_stars"))

        self.logger.info("Rendering completed")

    def save_results(self):
        """Save simulation results to a file."""
        self.logger.info("Saving propagation results")

        with open(str(self.res_dir / "PositionHistory.txt"), "w+") as file:
            for (date, sc_pos, sssb_pos) in zip(self.spacecraft.date_history,
                                                self.spacecraft.pos_history,
                                                self.sssb.pos_history):

                sc_pos = np.asarray(sc_pos.toArray())
                sssb_pos = np.asarray(sssb_pos.toArray())

                file.write(str(date) + "\t" + str(sssb_pos) + "\t"
                           + str(sc_pos) + "\n")

        self.logger.info("Propagation results saved")

if __name__ == "__main__":
    env = Environment("Didymos", 2 * 60.)
    env.simulate()
