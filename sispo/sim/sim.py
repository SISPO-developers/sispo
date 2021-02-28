"""Trajectory simulation and object rendering module."""

import json
import logging
from datetime import datetime
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
from org.orekit.time import (
    AbsoluteDate,
    TimeScalesFactory
)  # pylint: disable=import-error
from org.orekit.frames import FramesFactory  # pylint: disable=import-error
from org.orekit.utils import (
    Constants,
    PVCoordinates,
    AngularCoordinates
)  # pylint: disable=import-error
from org.hipparchus.geometry.euclidean.threed import (
    Vector3D,
    Rotation,
    RotationOrder,
    RotationConvention
)  # pylint: disable=import-error

from . import cb, sc, sssb, utilities
from .cb import *
from .sc import *
from .sssb import *

logger = logging.getLogger(__name__)

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
                 oneshot=False,
                 spacecraft=None,
                 opengl_renderer=False):

        self.opengl_renderer = opengl_renderer
        self.brdf_params = sssb.get('brdf_params', None)

        self.root_dir = Path(__file__).parent.parent.parent
        data_dir = self.root_dir / "data"
        self.models_dir = utilities.check_dir(data_dir / "models")

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

        # # Setup rendering engine (renderer)
        # self.setup_renderer()

        # Setup SSSB
        self.setup_sssb(sssb)

        # Setup SC
        self.setup_spacecraft(spacecraft, oneshot=oneshot)

        if not self.opengl_renderer:
            # Setup Sun
            self.setup_sun(sun)

            # Setup Lightref
            self.setup_lightref(lightref)

        logger.debug("Init finished")

    def setup_renderer(self):
        """Create renderer, apply common settings and create sc cam."""

        render_dir = utilities.check_dir(self.res_dir)
        raw_dir = utilities.check_dir(render_dir / "raw")

        if self.opengl_renderer:
            from .opengl import rendergl
            self.renderer = rendergl.RenderController(render_dir, stardb_path=self.starcat_dir)
            self.renderer.create_scene("SssbOnly")
        else:
            from .render import BlenderController
            self.renderer = BlenderController(render_dir,
                                              raw_dir,
                                              self.starcat_dir,
                                              self.inst,
                                              self.sssb_settings,
                                              self.with_infobox,
                                              self.with_clipping)

        self.renderer.create_camera("ScCam")

        self.renderer.configure_camera("ScCam", 
                                       self.inst.focal_l,
                                       self.inst.chip_w)

        if not self.opengl_renderer:
            self.renderer.create_scene("SssbConstDist")
            self.renderer.create_camera("SssbConstDistCam", scenes="SssbConstDist")
            self.renderer.configure_camera("SssbConstDistCam",
                                           self.inst.focal_l,
                                           self.inst.chip_w)

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
                'sispo_cam': self.inst,         # use sispo cam model instead 
                                                #of own (could use own if can give exposure & gain)
                'stars': True,                  # use own star db
                'lens_effects': False,          # includes the sun
                'brdf_params': self.brdf_params,
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
        # self.sun.render_obj = self.renderer.load_object(self.sun.model_file,
        #                                                 self.sun.name)

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
        # self.sssb.render_obj = self.renderer.load_object(
        #                             self.sssb.model_file, 
        #                             settings["model"]["name"], 
        #                             ["SssbOnly"] + ([] if self.opengl_renderer else ["SssbConstDist"]))
        # self.sssb.render_obj.rotation_mode = "AXIS_ANGLE"
        # self.sssb.render_obj.location = (0.0, 0.0, 0.0)

        # Setup previously generated coma
        coma = settings.get('coma', None)
        if coma:
            with open(coma['file'], 'r') as fh:
                coma.update(json.load(fh))
            assert self.opengl_renderer, '"coma" setting under "sssb" is currently only supported for opengl rendering'
            sssb_rot = coma.get('sssb_rot', False)
            if not sssb_rot:
                # if param missing and one shot mode, assumes that coma is created with same asteroid orientation
                assert self.sssb.rot_history, 'SSSB rotation state for cached coma is not given with "sssb_rot"'
                sssb_rot = self.sssb.rot_history[0]
                sssb_rot = (sssb_rot.getAngle(), *sssb_rot.getAxis(RotationConvention.FRAME_TRANSFORM).toArray())

            self.sssb.coma = self.renderer.load_coma(
                coma['tiles_file'],
                coma['dimensions'],
                coma['resolution'],
                coma.get('intensity', 1e-4),
                sssb_rot
            )

    def setup_spacecraft(self, spacecraft=None, oneshot=False):
        """Create Spacecraft and respective blender object."""

        sc_state = None
        sc_rot_state = None
        if spacecraft is None:
            sssb_state = self.sssb.get_state(self.encounter_date)
            sc_state = Spacecraft.calc_encounter_state(sssb_state,
                                                       self.minimum_distance,
                                                       self.relative_velocity,
                                                       self.with_terminator,
                                                       self.with_sunnyside)
        else:
            if 'r' in spacecraft:
                sc_state = PVCoordinates(Vector3D(spacecraft['r']), 
                                         Vector3D(spacecraft.get('v', [0.0, 0.0, 0.0])))
            if 'angleaxis' in spacecraft:
                sc_pxpz_rot = Rotation(Vector3D(spacecraft['angleaxis'][1:4]),
                                       spacecraft['angleaxis'][0],
                                       RotationConvention.FRAME_TRANSFORM)

                # transform camera where +x is forward and +z is up into -z is forward and +y is up
                mzpy_rot = self.pxpz_to_mzpy(sc_pxpz_rot)
                sc_rot_state = AngularCoordinates(mzpy_rot, Vector3D(0., 0., 0.))

        self.spacecraft = Spacecraft("CI",
                                     self.mu_sun,
                                     sc_state,
                                     self.encounter_date,
                                     rot_state=sc_rot_state,
                                     oneshot=oneshot)

    @staticmethod
    def pxpz_to_mzpy(pxpz_rot):
        pxpz_to_mzpy_rot = Rotation(0.5, 0.5, -0.5, -0.5, False)
        return pxpz_to_mzpy_rot.applyTo(pxpz_rot)

    @staticmethod
    def mzpy_to_pxpz(mzpy_rot):
        pxpz_to_mzpy_rot = Rotation(0.5, 0.5, -0.5, -0.5, False)
        return pxpz_to_mzpy_rot.applyInverseTo(mzpy_rot)

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
            raise SimulationError("Given lightref model filename does not exist.")

        # self.lightref = self.renderer.load_object(lightref_model_file,
        #                                           settings["model"]["name"],
        #                                           scenes="LightRef")
        # self.lightref.location = (0.0, 0.0, 0.0)

    def simulate(self):
        """Do simulation."""
        logger.debug("Starting simulation")

        logger.debug("Propagating SSSB")
        self.sssb.propagate(self.start_date,
                            self.end_date,
                            self.frames,
                            self.timesampler_mode,
                            self.slowmotion_factor)

        logger.debug("Propagating Spacecraft")
        self.spacecraft.propagate(self.start_date,
                                  self.end_date,
                                  self.frames,
                                  self.timesampler_mode,
                                  self.slowmotion_factor)

        logger.debug("Simulation completed")
        self.save_results()

    def render(self):
        """Render simulation scenario."""
        logger.debug("Rendering simulation")
        scaling = 1. if self.opengl_renderer else 1000.
        N = len(self.spacecraft.date_history)

        # Render frame by frame
        print("Rendering in progress...")
        for i, (date, sc_pos, sc_rot, sssb_pos, sssb_rot) in enumerate(zip(
                                                                   self.spacecraft.date_history,
                                                                   self.spacecraft.pos_history,
                                                                   self.spacecraft.rot_history,
                                                                   self.sssb.pos_history,
                                                                   self.sssb.rot_history)):

            date_str = datetime.strptime(date.toString(), "%Y-%m-%dT%H:%M:%S.%f")
            date_str = date_str.strftime("%Y-%m-%dT%H%M%S-%f")

            # metadict creation
            metainfo = dict()
            metainfo["sssb_pos"] = np.asarray(sssb_pos.toArray())
            metainfo["sc_pos"] = np.asarray(sc_pos.toArray())
            metainfo["distance"] = sc_pos.distance(sssb_pos)
            metainfo["date"] = date_str

            # Set Rotation
            angle, axis = convert_rot_to_angle_axis(sssb_rot, RotationConvention.FRAME_TRANSFORM)
            self.renderer.set_object_rot(angle, axis , self.sssb.render_obj)

            # Update environment
            # Removed unnecessary conditional, opengl can omit the scaling
            self.renderer.set_sun_location(-np.asarray(sssb_pos.toArray()), 
                                            scaling, getattr(self,"sun", None))

            # Update sssb and spacecraft
            pos_sc_rel_sssb = np.asarray(sc_pos.subtract(sssb_pos).toArray()) / scaling
            self.renderer.set_camera_location("ScCam", pos_sc_rel_sssb)
            if self.spacecraft.auto_targeting:
                self.renderer.target_camera(self.sssb.render_obj, "ScCam")
            else:
                angle, axis = convert_rot_to_angle_axis(sc_rot, RotationConvention.FRAME_TRANSFORM)
                self.renderer.set_camera_rot(angle, axis, "ScCam")

            if not self.opengl_renderer:
                # Update scenes/cameras
                pos_cam_const_dist = pos_sc_rel_sssb * scaling / np.sqrt(
                                        np.dot(pos_sc_rel_sssb, pos_sc_rel_sssb))
                self.renderer.set_camera_location("SssbConstDistCam", pos_cam_const_dist)
                self.renderer.target_camera(self.sssb.render_obj, "SssbConstDistCam")

                lightrefcam_pos = -np.asarray(sssb_pos.toArray()) * scaling \
                                  / np.sqrt(np.dot(np.asarray(sssb_pos.toArray()), 
                                    np.asarray(sssb_pos.toArray())))
                self.renderer.set_camera_location("LightRefCam", lightrefcam_pos)
                self.renderer.target_camera(self.sun.render_obj, "CalibrationDisk")
                self.renderer.target_camera(self.lightref, "LightRefCam")

            # Render blender scenes
            self.renderer.render(metainfo)

            print('%d/%d' % (i+1, N))

        logger.debug("Rendering completed")

    def save_results(self):
        """Save simulation results to a file."""
        logger.debug("Saving propagation results")

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

        logger.debug("Propagation results saved")


def convert_rot_to_angle_axis(rot, rot_conv):
    angle = rot.getAngle()
    axis = np.array(rot.getAxis(rot_conv).toArray())

    return angle, axis 



def blend_to_icrf(sssb_cam_r, sssb_light_r, cam_quat, dist=1.5e11, verbose=False):
    """
    transform manually found (using blender) relative pose and light direction
    to sc and sssb params in input definition file:

    ...
    "spacecraft":
    {
        "r": sun_sc_r,
        "angleaxis": sc_rot
    },
    "sssb":
    {
        "traj": {
            "r": sun_sssb_r,
            ...
        }
        ...
    }
    ...
    """
    sssb_light_r = np.array(sssb_light_r)
    sun_sssb_r = -sssb_light_r * (dist/np.linalg.norm(sssb_light_r))
    sun_sc_r = sun_sssb_r + np.array(sssb_cam_r) * 1000     # km shows in meters in blender
    sc_rot = Rotation(*cam_quat, False)
    icrf2gl_rot = Rotation(0.5, 0.5, -0.5, -0.5, False)
#    sc_rot = icrf2gl_rot.applyTo(sc_rot)
    sc_rot = icrf2gl_rot.revert().applyTo(sc_rot)        # == sc_q * icrf2gl_q.conj()
    #icrf2gl_rot.applyTo(sc_icrf_rot)            # == sc_icrf_q * icrf2gl_q

    if verbose:
        angle = sc_rot.getAngle()
        axis = sc_rot.getAxis(RotationConvention.FRAME_TRANSFORM)
        sc_rot_aa = np.array([angle, *axis.toArray()])
        arr2str = lambda arr: '[%s]' % ', '.join(['%f' % v for v in arr])
        logger.debug('sc angle-axis: %s' % arr2str(sc_rot_aa))
        logger.debug('sun-sc vect: %s' % arr2str(sun_sc_r))
        logger.debug('sun-sssb vect: %s' % arr2str(sun_sssb_r))

    return sc_rot, sun_sc_r, sun_sssb_r,


if __name__ == "__main__":
    pass
