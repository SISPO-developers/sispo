"""Trajectory simulation and object rendering module."""

from datetime import datetime
import json
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
from simulation.sc import Instrument, Spacecraft
from simulation.sssb import SmallSolarSystemBody
import simulation.render as render
import simulation.compositor as compositor
import utils


class SimulationError(RuntimeError):
    """Generic simulation error."""
    pass


class Environment():
    """Simulation environment."""

    def __init__(self, settings):

        self.settings = settings

        self.name = settings["name"]
        
        self.root_dir = Path(__file__).parent.parent.parent
        data_dir = self.root_dir / "data"
        self.models_dir = utils.check_dir(data_dir / "models")

        self.res_dir = utils.check_dir(data_dir / "results" / self.name)

        self.inst = Instrument(settings["instrument"])
        #comp = compositor.ImageCompositor(self.res_dir, self.inst)

        self.logger = utils.create_logger("simulation")

        self.ts = TimeScalesFactory.getTDB()
        self.ref_frame = FramesFactory.getICRF()
        self.mu_sun = Constants.IAU_2015_NOMINAL_SUN_GM

        encounter_date = settings["encounter_date"]
        self.encounter_date = AbsoluteDate(int(encounter_date["year"]),
                                           int(encounter_date["month"]),
                                           int(encounter_date["day"]),
                                           int(encounter_date["hour"]),
                                           int(encounter_date["minutes"]),
                                           float(encounter_date["seconds"]),
                                           self.ts)
        self.duration = settings["duration"]
        self.start_date = self.encounter_date.shiftedBy(-self.duration / 2.)
        self.end_date = self.encounter_date.shiftedBy(self.duration / 2.)

        self.frames = settings["frames"]

        self.minimum_distance = settings["encounter_dist"]
        self.with_terminator = bool(settings["with_terminator"])
        self.with_sunnyside = bool(settings["with_sunnyside"])
        self.timesampler_mode = settings["timesampler_mode"]
        self.slowmotion_factor = settings["slowmotion_factor"]

        self.render_settings = dict()
        self.render_settings["exposure"] = settings["exposure"]
        self.render_settings["samples"] = settings["samples"]
        self.render_settings["device"] = settings["device"]
        self.render_settings["tile"] = settings["tile"]

        # Setup rendering engine (renderer)
        self.setup_renderer()

        # Setup Sun
        self.setup_sun(settings["sun"])

        # Setup SSSB
        self.setup_sssb(settings["sssb"])

        # Setup SC
        self.setup_spacecraft()

        # Setup Lightref
        self.setup_lightref(settings["lightref"])

    def setup_renderer(self):
        """Create renderer, apply common settings and create sc cam."""

        self.render_dir = utils.check_dir(self.res_dir / "rendering")

        self.renderer = render.BlenderController(self.render_dir)
        self.renderer.create_camera("ScCam")
        self.renderer.configure_camera("ScCam", self.inst.focal_l, self.inst.chip_w)

        self.renderer.create_scene("SssbConstDist")
        self.renderer.create_camera("SssbConstDistCam", scenes="SssbConstDist")
        self.renderer.configure_camera("SssbConstDistCam", self.inst.focal_l, self.inst.chip_w)

        self.renderer.create_scene("LightRef")
        self.renderer.create_camera("LightRefCam", scenes="LightRef")
        self.renderer.configure_camera("LightRefCam", self.inst.focal_l, self.inst.chip_w)

        self.renderer.set_device(self.render_settings["device"])
        self.renderer.set_samples(self.render_settings["samples"])
        self.renderer.set_exposure(self.render_settings["exposure"])
        self.renderer.set_resolution(self.inst.res)
        self.renderer.set_output_format()

    def setup_sun(self, settings):
        """Create Sun and respective render object."""
        sun_model_file = Path(settings["model"]["file"])

        try:
            sun_model_file = sun_model_file.resolve()
        except OSError as e:
            raise SimulationError(e)

        if not sun_model_file.is_file():
                sun_model_file = self.models_dir / sun_model_file.name
                sun_model_file = sun_model_file.resolve()
        
        if not sun_model_file.is_file():
            raise SimulationError("Given SSSB model filename does not exist.")

        self.sun = CelestialBody(settings["model"]["name"],
                                 model_file=sun_model_file)
        self.sun.render_obj = self.renderer.load_object(self.sun.model_file,
                                                        self.sun.name)

    def setup_sssb(self, settings):
        """Create SmallSolarSystemBody and respective blender object."""
        sssb_model_file = Path(settings["model"]["file"])

        try:
            sssb_model_file = sssb_model_file.resolve()
        except OSError as e:
            raise SimulationError(e)

        if not sssb_model_file.is_file():
                sssb_model_file = self.models_dir / sssb_model_file.name
                sssb_model_file = sssb_model_file.resolve()
        
        if not sssb_model_file.is_file():
            raise SimulationError("Given SSSB model filename does not exist.")

        self.sssb = SmallSolarSystemBody(settings["model"]["name"],
                                          self.mu_sun, 
                                          settings["trj"],
                                          settings["att"],
                                          model_file=sssb_model_file)
        self.sssb.render_obj = self.renderer.load_object(self.sssb.model_file,
                                                         settings["model"]["name"],
                                                         ["SssbOnly", 
                                                          "SssbConstDist"])
        self.sssb.render_obj.rotation_mode = "AXIS_ANGLE"

    def setup_spacecraft(self):
        """Create Spacecraft and respective blender object."""
        sssb_state = self.sssb.get_state(self.encounter_date)
        sc_state = Spacecraft.calc_encounter_state(sssb_state,
                                                   self.minimum_distance,
                                                   self.with_terminator,
                                                   self.with_sunnyside)
        self.spacecraft = Spacecraft("CI", 
                                     self.mu_sun,
                                     sc_state,
                                     self.encounter_date)

    def setup_lightref(self, settings):
        """Create lightreference blender object."""
        lightref_model_file = Path(settings["model"]["file"])

        try:
            lightref_model_file = lightref_model_file.resolve()
        except OSError as e:
            raise SimulationError(e)

        if not lightref_model_file.is_file():
                lightref_model_file = self.models_dir / lightref_model_file.name
                lightref_model_file = lightref_model_file.resolve()
        
        if not lightref_model_file.is_file():
            raise SimulationError("Given SSSB model filename does not exist.")

        self.lightref = self.renderer.load_object(lightref_model_file,
                                                  settings["model"]["name"],
                                                  scenes="LightRef")
        self.lightref.location = (0, 0, 0)

    def simulate(self):
        """Do simulation."""
        self.logger.info("Starting simulation")

        self.logger.info("Propagating SSSB")
        self.sssb.propagate(self.start_date,
                            self.end_date,
                            self.frames,
                            self.timesampler_mode,
                            self.slowmotion_factor)

        self.logger.info("Propagating Spacecraft")
        self.spacecraft.propagate(self.start_date,
                                  self.end_date,
                                  self.frames,
                                  self.timesampler_mode,
                                  self.slowmotion_factor)

        self.logger.info("Simulation completed")
        self.save_results()

    def render(self):
        """Render simulation scenario."""
        self.logger.info("Rendering simulation")

        # Render frame by frame
        for (date, sc_pos, sssb_pos, sssb_rot) in zip(self.spacecraft.date_history,
                                                      self.spacecraft.pos_history,
                                                      self.sssb.pos_history,
                                                      self.sssb.rot_history):

            date_str = datetime.strptime(date.toString(), "%Y-%m-%dT%H:%M:%S.%f")
            date_str = date_str.strftime("%Y-%m-%dT%H%M%S-%f")

            # Update environment
            self.sun.render_obj.location = -np.asarray(sssb_pos.toArray()) / 1000.

            # Update sssb and spacecraft
            pos_sc_rel_sssb = np.asarray(sc_pos.subtract(sssb_pos).toArray()) / 1000.
            self.renderer.set_camera_location("ScCam", pos_sc_rel_sssb)            

            sssb_axis = sssb_rot.getAxis(self.sssb.rot_conv)
            sssb_angle = sssb_rot.getAngle()
            self.sssb.render_obj.rotation_axis_angle = (sssb_angle, sssb_axis.x, sssb_axis.y, sssb_axis.z)          

            self.renderer.target_camera(self.sssb.render_obj, "ScCam")
            
            # Update optional scenes/cameras
            pos_cam_const_dist = pos_sc_rel_sssb * 1000. / np.sqrt(np.dot(pos_sc_rel_sssb, pos_sc_rel_sssb))
            self.renderer.set_camera_location("SssbConstDistCam", pos_cam_const_dist)
            self.renderer.target_camera(self.sssb.render_obj, "SssbConstDistCam")

            lightrefcam_pos = -np.asarray(sssb_pos.toArray()) * 1000. /np.sqrt(np.dot(np.asarray(sssb_pos.toArray()),np.asarray(sssb_pos.toArray())))
            self.renderer.set_camera_location("LightRefCam", lightrefcam_pos)
            self.renderer.target_camera(self.sun.render_obj, "CalibrationDisk")
            self.renderer.target_camera(self.lightref, "LightRefCam")
            
            # Render blender scenes
            self.renderer.render(date_str)

            # Render star background
            fluxes = self.renderer.render_starmap(self.inst.res, date_str)

            metadict = dict()
            metadict["sssb_pos"] = np.asarray(sssb_pos.toArray())
            metadict["sc_pos"] = np.asarray(sc_pos.toArray())
            metadict["distance"] = sc_pos.distance(sssb_pos)
            metadict["date"] = date_str
            metadict["sc_rel_pos"] = pos_sc_rel_sssb
            metadict["total_flux"] = fluxes[0]

            self.write_meta_file(date_str, metadict)

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

    def write_meta_file(self, suffix, metadict):
        """Writes metafile for a frame."""

        filename = self.render_dir / ("Metadata_" + str(suffix))
        filename = str(filename)

        file_extension = ".json"
        if filename[-len(file_extension):] != file_extension:
            filename += file_extension

        with open(filename, "w+") as metafile:
            json.dump(metadict, metafile, default=utils.serialise)

if __name__ == "__main__":
    env = Environment("Didymos", 2 * 60.)
    env.simulate()
