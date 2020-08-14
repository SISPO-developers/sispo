"""Trajectory simulation and object rendering module."""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import orekit
from orekit.pyhelpers import setup_orekit_curdir
#################### orekit VM init ####################
FILE_DIR = Path(__file__).parent.resolve()
OREKIT_DATA_FILE = FILE_DIR / "orekit-data.zip"
OREKIT_VM = orekit.initVM() # pylint: disable=no-member
setup_orekit_curdir(str(OREKIT_DATA_FILE))
#################### orekit VM init ####################
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # pylint: disable=import-error
from org.orekit.frames import FramesFactory  # pylint: disable=import-error
from org.orekit.utils import Constants  # pylint: disable=import-error

from mathutils import Matrix

from . import cb
from .cb import *
from . import sc
from .sc import *
from . import sssb
from .sssb import *
from . import utils

import  mathutils

class SimulationError(RuntimeError):
    """Generic simulation error."""
    pass


class Environment():
    """
    Simulation environment.

    This environment is used to propagate trajectories and render images at
    each simulation step.
    """

    def __init__(self,
                 res_dir,
                 starcat_dir,
                 instrument,
                 with_infobox,
                 with_clipping,
                 sssb,
                 sun,
                 lightref,
                 encounter_date,
                 duration,
                 frames,
                 encounter_distance,
                 relative_velocity,
                 with_sunnyside,
                 with_terminator,
                 timesampler_mode,
                 slowmotion_factor,
                 exposure,
                 samples,
                 device,
                 tile_size,
                 ext_logger=None,
                 opengl_renderer=False,
                 refl_model_params=None):

        if ext_logger is not None:
            self.logger = ext_logger
        else:
            self.logger = utils.create_logger()

        self.opengl_renderer = opengl_renderer
        self.refl_model_params = refl_model_params  # TODO: object specific refl_model_params, read from object files

        self.root_dir = Path(__file__).parent.parent.parent
        data_dir = self.root_dir / "data"
        self.models_dir = utils.check_dir(data_dir / "models")

        self.res_dir = res_dir
        
        self.starcat_dir = starcat_dir

        self.inst = Instrument(instrument)

        self.ts = TimeScalesFactory.getTDB()
        self.ref_frame = FramesFactory.getICRF()
        self.mu_sun = Constants.IAU_2015_NOMINAL_SUN_GM

        encounter_date = encounter_date
        self.encounter_date = AbsoluteDate(int(encounter_date["year"]),
                                           int(encounter_date["month"]),
                                           int(encounter_date["day"]),
                                           int(encounter_date["hour"]),
                                           int(encounter_date["minutes"]),
                                           float(encounter_date["seconds"]),
                                           self.ts)
        self.duration = duration
        self.start_date = self.encounter_date.shiftedBy(-self.duration / 2.)
        self.end_date = self.encounter_date.shiftedBy(self.duration / 2.)

        self.frames = frames

        self.minimum_distance = encounter_distance
        self.relative_velocity = relative_velocity
        self.with_terminator = bool(with_terminator)
        self.with_sunnyside = bool(with_sunnyside)
        self.timesampler_mode = timesampler_mode
        self.slowmotion_factor = slowmotion_factor

        self.render_settings = dict()
        self.render_settings["exposure"] = exposure
        self.render_settings["samples"] = samples
        self.render_settings["device"] = device
        self.render_settings["tile"] = tile_size

        self.sssb_settings = sssb
        self.with_infobox = with_infobox
        self.with_clipping = with_clipping

        # Setup rendering engine (renderer)
        self.setup_renderer()

        # Setup SSSB
        self.setup_sssb(sssb)

        # Setup SC
        self.setup_spacecraft()

        if not self.opengl_renderer:
            # Setup Sun
            self.setup_sun(sun)

            # Setup Lightref
            self.setup_lightref(lightref)

    def setup_renderer(self):
        """Create renderer, apply common settings and create sc cam."""

        render_dir = utils.check_dir(self.res_dir)
        raw_dir = utils.check_dir(render_dir / "raw")

        if self.opengl_renderer:
            from .opengl import rendergl
            self.renderer = rendergl.RenderController(render_dir, stardb_path=self.starcat_dir, logger=self.logger)
            self.renderer.create_scene("SssbOnly")
        else:
            from .render import BlenderController
            self.renderer = BlenderController(render_dir,
                                              raw_dir,
                                              self.starcat_dir,
                                              self.inst,
                                              self.sssb_settings,
                                              self.with_infobox,
                                              self.with_clipping,
                                              ext_logger=self.logger)

        self.renderer.create_camera("ScCam")

        self.renderer.configure_camera("ScCam", 
                                       self.inst.focal_l,
                                       self.inst.chip_w)

        self.renderer.create_scene("SssbConstDist")
        self.renderer.create_camera("SssbConstDistCam", scenes="SssbConstDist")
        self.renderer.configure_camera("SssbConstDistCam", 
                                       self.inst.focal_l,
                                       self.inst.chip_w)

        if not self.opengl_renderer:
            self.renderer.create_scene("LightRef")
            self.renderer.create_camera("LightRefCam", scenes="LightRef")
            self.renderer.configure_camera("LightRefCam",
                                           self.inst.focal_l,
                                           self.inst.chip_w)
        else:
            # as use sispo cam model
            self.renderer.set_scene_config({
                'debug': False,
                'flux_only': False,
                'sispo_cam': self.inst,         # use sispo cam model instead of own (could use own if can give exposure & gain)
                'stars': True,                  # use own star db
                'lens_effects': False,          # includes the sun
                'hapke_params': self.refl_model_params or rendergl.RenderObject.HAPKE_PARAMS,
                # TODO: object specific refl_model_params, read from object files
            })

        self.renderer.set_device(self.render_settings["device"], 
                                 self.render_settings["tile"])
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
        self.sssb.render_obj.location = (0, 0, 0)

    def setup_spacecraft(self):
        """Create Spacecraft and respective blender object."""
        sssb_state = self.sssb.get_state(self.encounter_date)
        sc_state = Spacecraft.calc_encounter_state(sssb_state,
                                                   self.minimum_distance,
                                                   self.relative_velocity,
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
        self.logger.debug("Starting simulation")

        self.logger.debug("Propagating SSSB")
        self.sssb.propagate(self.start_date,
                            self.end_date,
                            self.frames,
                            self.timesampler_mode,
                            self.slowmotion_factor)

        self.logger.debug("Propagating Spacecraft")
        self.spacecraft.propagate(self.start_date,
                                  self.end_date,
                                  self.frames,
                                  self.timesampler_mode,
                                  self.slowmotion_factor)

        self.logger.debug("Simulation completed")
        self.save_results()


    def set_rotation(self, sssb_rot, sispoObj):
        """Set rotation and returns original transformation matrix"""
        """Assumes that the blender scaling is set to 1"""
        #sssb_axis = sssb_rot.getAxis(self.sssb.rot_conv)
        sssb_angle = sssb_rot.getAngle()
        
        eul0 = mathutils.Euler((0.0, 0.0, sssb_angle), 'XYZ')
        eul1 = mathutils.Euler((0.0, sispoObj.Dec, 0.0), 'XYZ')
        eul2 = mathutils.Euler((0.0, 0.0, sispoObj.RA), 'XYZ')

        R0 = eul0.to_matrix()
        R1 = eul1.to_matrix()
        R2 = eul2.to_matrix()

        try:
            M = R2 @ R1 @ R0
        except TypeError:
            # earlier versions of mathutils (e.g. 2.78) does not yet support @ for matrix multiplication
            M = R2 * R1 * R0

        original_transform = sispoObj.render_obj.matrix_world

        sispoObj.render_obj.matrix_world = M.to_4x4()
        return original_transform


    def render(self):
        """Render simulation scenario."""
        self.logger.debug("Rendering simulation")
        scaling = 1 if self.opengl_renderer else 1000.

        # Render frame by frame
        for (date, sc_pos, sssb_pos, sssb_rot) in zip(self.spacecraft.date_history,
                                                      self.spacecraft.pos_history,
                                                      self.sssb.pos_history,
                                                      self.sssb.rot_history):

            date_str = datetime.strptime(date.toString(), "%Y-%m-%dT%H:%M:%S.%f")
            date_str = date_str.strftime("%Y-%m-%dT%H%M%S-%f")

            # metadict creation
            metainfo = dict()
            metainfo["sssb_pos"] = np.asarray(sssb_pos.toArray())
            metainfo["sc_pos"] = np.asarray(sc_pos.toArray())
            metainfo["distance"] = sc_pos.distance(sssb_pos)
            metainfo["date"] = date_str

            orig_transform = self.set_rotation(sssb_rot, self.sssb)

            # Update environment
            if not self.opengl_renderer:
                self.sun.render_obj.location = -np.asarray(sssb_pos.toArray()) / scaling
            else:
                self.renderer.set_sun_location(-np.asarray(sssb_pos.toArray()))

            # Update sssb and spacecraft
            pos_sc_rel_sssb = np.asarray(sc_pos.subtract(sssb_pos).toArray()) / scaling
            self.renderer.set_camera_location("ScCam", pos_sc_rel_sssb)
            self.renderer.target_camera(self.sssb.render_obj, "ScCam")
            
            # Update scenes/cameras
            pos_cam_const_dist = pos_sc_rel_sssb * scaling / np.sqrt(np.dot(pos_sc_rel_sssb, pos_sc_rel_sssb))
            self.renderer.set_camera_location("SssbConstDistCam", pos_cam_const_dist)
            self.renderer.target_camera(self.sssb.render_obj, "SssbConstDistCam")

            if not self.opengl_renderer:
                lightrefcam_pos = -np.asarray(sssb_pos.toArray()) * scaling \
                                  / np.sqrt(np.dot(np.asarray(sssb_pos.toArray()), np.asarray(sssb_pos.toArray())))
                self.renderer.set_camera_location("LightRefCam", lightrefcam_pos)
                self.renderer.target_camera(self.sun.render_obj, "CalibrationDisk")
                self.renderer.target_camera(self.lightref, "LightRefCam")

            # Render blender scenes
            self.renderer.render(metainfo)
            
            #set original rotation
            self.sssb.render_obj.matrix_world = orig_transform
                
        self.logger.debug("Rendering completed")

    def save_results(self):
        """Save simulation results to a file."""
        self.logger.debug("Saving propagation results")

        float_formatter = "{:.16f}".format
        np.set_printoptions(formatter={'float_kind': float_formatter})
        vec2str = lambda v: str(np.asarray(v.toArray()))

        print_list = [
            [str(v) for v in self.spacecraft.date_history],
            [vec2str(v) for v in self.spacecraft.pos_history],
            [vec2str(v) for v in self.spacecraft.vel_history],
            #[vec2str(v) for v in self.spacecraft.rot_history],
            [vec2str(v) for v in self.sssb.pos_history],
            [vec2str(v) for v in self.sssb.vel_history],
            #[vec2str(v) for v in self.sssb.rot_history],
        ]

        with open(str(self.res_dir / "DynamicsHistory.txt"), "w+") as file:
            for i in range(len(self.spacecraft.date_history)):
                line = "\t".join([v[i] for v in print_list]) + "\n"
                file.write(line)

        self.logger.debug("Propagation results saved")


if __name__ == "__main__":
    pass
