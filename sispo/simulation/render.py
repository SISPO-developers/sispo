"""
Module providing an interface to a rendering engine (renderer).
   
This implementation uses the blender python module bpy.
"""

import math
from pathlib import Path
import struct
import time
import zlib

import bpy
from mathutils import Vector # pylint: disable=import-error
import numpy as np
import OpenEXR
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean

import utils

logger = utils.create_logger("rendering")

class RenderingError(RuntimeError):
    """Generic error for rendering process."""
    pass


class BlenderControllerError(RuntimeError):
    """Generic error for BlenderController."""
    pass


class BlenderController:
    """Class to control blender module behaviour."""

    def __init__(self, render_dir):
        """Initialise blender controller class."""

        self.res_dir = render_dir
        self.cycles = bpy.context.preferences.addons["cycles"]

        self.default_scene = bpy.context.scene
        self.scenes = bpy.data.scenes
        self.cameras = bpy.data.cameras

        # Initial scene is MainScene, clear from objects, and set defaults
        self.default_scene.name = "MainScene"
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj)
        self.set_scene_defaults(self.default_scene)

        self.set_device()

        self.render_id = zlib.crc32(struct.pack("!f", time.time()))

    def create_scene(self, scene_name):
        """Add empty scene."""
        bpy.ops.scene.new(type="FULL_COPY")
        bpy.context.scene.name = scene_name

        self.set_scene_defaults(scene_name)

        bpy.context.window.scene = self.default_scene

    def set_scene_defaults(self, scenes=None):
        """Sets default settings to a scene."""
        for scene in self._get_scenes_iter(scenes):
            scene.render.image_settings.color_mode = "RGBA"
            scene.render.image_settings.use_zbuffer = True
            scene.render.resolution_percentage = 100 # TODO: change, 5 is debug setting
            scene.view_settings.view_transform = "Raw"
            scene.view_settings.look = "None"
        
            scene.render.engine = "CYCLES"
            scene.cycles.feature_set = "EXPERIMENTAL"
            scene.cycles.min_bounces = 3
            scene.cycles.max_bounces = 128
            scene.cycles.caustics_reflective = True
            scene.cycles.caustics_refractive = True
            scene.cycles.diffuse_bounces = 128
            scene.cycles.glossy_bounces = 128
            scene.cycles.transmission_bounces = 128
            scene.cycles.volume_bounces = 128
            scene.cycles.transparent_min_bounces = 8
            scene.cycles.transparent_max_bounces = 128
            scene.cycles.use_square_samples = True
            #scene.cycles.use_animated_seed = True
            scene.cycles.seed = time.time()
            scene.cycles.film_transparent = True

    def set_device(self, device="AUTO", scenes=None):
        """Set cycles rendering device for given scenes.

        When device="AUTO" it is attempted to use GPU first, otherwise
        fallback is CPU. Currently, assumes set_device is only used once.
        """
        logger.info("Attempting to set cycle rendering device to: %s", device)

        self.device = self._determine_device(device)
        self._set_cycles_device()
        tile_size = self.get_tile_size()

        # Sets render device of scenes
        for scene in self._get_scenes_iter(scenes):
            scene.cycles.device = self.device
            scene.render.tile_x = tile_size
            scene.render.tile_y = tile_size


    def _determine_device(self, device):
        """Determines the render device based on availability and input.

        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        needs to be called otherwise .devices collection is not initialised.
        """
        self.cycles.preferences.get_devices()
        devices = self.cycles.preferences.devices
        logger.info("Available devices: %s", [dev.name for dev in devices])

        if device in ("AUTO", "GPU"):
            device_types = {device.type for device in devices}

            if "CUDA" in device_types:
                used_device = "GPU"
            else:
                used_device = "CPU"

        elif device == "CPU":
            used_device = "CPU"

        else:
            logger.info("Invalid rendering device setting.")
            raise BlenderControllerError("Invalid rendering device setting.")

        return used_device

    def _set_cycles_device(self):
        """Applies self.device setting to cycles itself."""
        devices = self.cycles.preferences.devices

        if self.device in ("CPU", "GPU"):
            if self.device == "GPU":
                device_type = "CUDA"
            else:
                device_type = "NONE"
            self.cycles.preferences.compute_device_type = device_type
            for device in devices:
                if device.type == device_type:
                    device.use = True
                    logger.info("%s device name: %s", device_type, device.name)
                else:
                    device.use = False
        else:
            logger.info("Invalid device: %s", self.device)
            raise BlenderControllerError(f"Invalid device: {self.device}")

    def get_tile_size(self):
        """Determine size of tiles while rendering based on render device."""
        if self.device == "GPU":
            tile_size = 512
        elif self.device == "CPU":
            tile_size = 128
        else:
            logger.info("Can not get tile size for device %s", self.device)
            raise BlenderControllerError("Can not get tile size.")

        return tile_size

    def set_samples(self, samples=6, scenes=None):
        """Set number of samples to render for each pixel."""
        for scene in self._get_scenes_iter(scenes):
            scene.cycles.samples = samples

    def set_exposure(self, exposure, scenes=None):
        """Set exposure value."""
        for scene in self._get_scenes_iter(scenes):
            scene.view_settings.exposure = exposure

    def set_resolution(self, res, scenes=None):
        """Sets resolution of rendered image."""
        res_x = res[0]
        res_y = res[1]

        for scene in self._get_scenes_iter(scenes):
            scene.render.resolution_x = res_x
            scene.render.resolution_y = res_y

    def set_output_format(self,
                          file_format="OPEN_EXR",
                          color_depth="32",
                          use_preview=True,
                          scenes=None):
        """Set output file format."""
        for scene in self._get_scenes_iter(scenes):
            scene.render.image_settings.file_format = file_format
            scene.render.image_settings.color_depth = color_depth
            scene.render.image_settings.use_preview = use_preview

    def set_output_file(self, name_suffix=None, scene=bpy.context.scene):
        """Set output file path to given scenes with prior extension check."""
        filename = self.res_dir / (scene.name + "_" + str(name_suffix))
        filename = str(filename)

        file_extension = ".exr"
        if filename[-4:] != file_extension:
            filename += file_extension

        scene.render.filepath = str(filename)

    def create_camera(self, camera_name="Camera", scenes=None):
        """Create new camera and add to relevant scenes."""
        cam = bpy.data.cameras.new(camera_name)
        camera = bpy.data.objects.new(camera_name, object_data=cam)
        camera.name = camera_name
        self.set_camera_location(camera_name, (0, 0, 0))

        for scene in self._get_scenes_iter(scenes):
            scene.camera = camera
            scene.collection.objects.link(camera)

    def configure_camera(self,
                         camera_name="Camera",
                         lens=35,
                         sensor=32,
                         clip_start=1E-5,
                         clip_end=1E32,
                         mode="PERSP", # Modes ORTHO, PERSP
                         ortho_scale=7):
        """Set camera configuration values."""
        camera = self.cameras[camera_name]      
        camera.clip_end = clip_end
        camera.clip_start = clip_start
        camera.lens = lens
        camera.ortho_scale = ortho_scale
        camera.sensor_width = sensor
        camera.type = mode

    def set_camera_location(self, camera_name="Camera", location=(0, 0, 0)):
        camera = bpy.data.objects[camera_name]
        camera.location = location

    def target_camera(self, target, camera_name="Camera"):
        """Target camera towards target."""
        camera = bpy.data.objects[camera_name]
        camera_constr = camera.constraints.new(type="TRACK_TO")
        camera_constr.track_axis = "TRACK_NEGATIVE_Z"
        camera_constr.up_axis = "UP_Y"
        camera_constr.target = target

    def update(self, scenes=None):
        """Update scenes."""
        for scene in self._get_scenes_iter(scenes):
            scene.cycles.seed = time.time()
            scene.view_layers.update()

    def render(self, name_suffix=None, scenes=None):
        """Render given scene."""
        if name_suffix is None:
            name = self.res_dir / f"r{self.render_id:0.8X}"

        for scene in self._get_scenes_iter(scenes):
            self.update(scene)
            self.set_output_file(name_suffix, scene)
            bpy.ops.render.render(write_still=True, scene=scene.name)
            self.save_blender_dfile(name_suffix, scene)

    def load_object(self, filename, object_name, scenes=None):
        """Load blender object from file."""
        filename = str(filename)

        with bpy.data.libraries.load(filename) as (data_from, data_to):
            data_to.objects = [
                name for name in data_from.objects if name == object_name]
        if data_to.objects:
            obj = data_to.objects[0]
            obj.animation_data_clear()
            
            for scene in self._get_scenes_iter(scenes):
                scene.collection.objects.link(obj)
            return obj
        else:
            msg = f"{object_name} not found in {filename}"
            logger.info(msg)
            raise BlenderControllerError(msg)

    def create_empty(self, name="Empty", scenes=None):
        """Create new, empty blender object."""
        obj_empty = bpy.data.objects.new(name, None)
        for scene in self._get_scenes_iter(scenes):
            scene.collection.objects.link(obj_empty)
        return obj_empty

    def save_blender_dfile(self, name_suffix=None, scene=bpy.context.scene):
        """Save a blender d file."""
        filename = self.res_dir / (scene.name + "_" + str(name_suffix))
        filename = str(filename)

        file_extension = ".blend"
        if filename[-6:] != file_extension:
            filename += file_extension

        bpy.ops.wm.save_as_mainfile(filepath=filename)

    def _get_scenes_iter(self, scenes):
        """Checks scenes input to allow different types and create iterator.
        
        Input can either be None, a scene name (str), a list of scene names,
        a single scene, or a list of scenes.
        Output is an iterator which can be used for looping through scenes.
        """
        if scenes is None:
            output = self.scenes
        elif isinstance(scenes, str):
            output = [self.scenes[scenes]]
        elif isinstance(scenes, bpy.types.Scene):
            output = [scenes]
        elif isinstance(scenes, list):
            if isinstance(scenes[0], str):
                output = []
                for scene_name in scenes:
                    output.append(self.scenes[scene_name])
            elif isinstance(scenes[0], bpy.types.Scene):
                output = scenes
            else:
                logger.info("Invalid scenes input %s", scenes)
                raise BlenderControllerError(f"Invalid scenes input {scenes}")
        else:
            logger.info("Invalid scenes input %s", scenes)
            raise BlenderControllerError(f"Invalid scenes input {scenes}")

        return iter(output)


    def render_starmap(self, stardata, fov_vecs, img_size, name_suffix):
        """Render a starmap from given data and field of view."""
        (direction, right_edge, _, upper_edge, _) = fov_vecs
        (res_x, res_y) = img_size

        upper_edge -= direction
        right_edge -= direction
        total_flux = 0.
        up_norm = upper_edge.normalized()
        right_norm = right_edge.normalized()
        f_over_h_ccd_2 = 1. / upper_edge.length
        f_over_w_ccd_2 = 1. / right_edge.length
        ss = 2
        starmap = np.zeros((res_y * ss, res_x * ss, 4), np.float32)
        # Add alpha channel
        starmap[:,:,3] = 1.

        for star in stardata:
            mag_star = star[2]
            flux = np.power(10., -0.4 * mag_star)
            total_flux += flux
            ra_star = np.radians(star[0])
            dec_star = np.radians(star[1])

            z_star = np.sin(dec_star)
            x_star = np.cos(dec_star) * np.cos(ra_star - np.pi)
            y_star = -np.cos(dec_star) * np.sin(ra_star - np.pi)
            vec = [x_star, y_star, z_star]
            vec2 = [x_star, -y_star, z_star]
            if np.dot(vec, direction) < np.dot(vec2, direction):
                vec = vec2
            x_pix = ss * (f_over_w_ccd_2 * np.dot(right_norm, vec) \
                    / np.dot(direction, vec) + 1.) * (res_x - 1) / 2.
            x_pix = min(round(x_pix), res_x * ss - 1)
            x_pix = max(0, int(x_pix))
            y_pix = ss * (-f_over_h_ccd_2 * np.dot(up_norm, vec) \
                    / np.dot(direction, vec) + 1.) * (res_y - 1) / 2.
            y_pix = min(round(y_pix), res_y * ss - 1)
            y_pix = max(0, int(y_pix))
            # Add flux to color channels
            starmap[y_pix, x_pix, 0:3] += flux

        sm_gauss = gaussian(starmap, ss / 2., multichannel=True)

        sm_scale = np.zeros((res_y, res_x, 4), np.float32)
        factors = (ss, ss)
        for c in range(4):
            sm_scale[:, :, c] = downscale_local_mean(sm_gauss[:, :, c], factors) * (ss * ss)

        filename = self.res_dir / ("Stars_" + name_suffix)
        write_openexr_image(filename, sm_scale)

        return (total_flux, np.sum(sm_scale[:, :, 0]))


def get_fov_vecs(camera_name, scene_name):
    """Get camera position and direction vectors."""
    camera = bpy.data.objects[camera_name]
    up_vec = camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
    direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    right_vec = direction.cross(up_vec)

    scene = bpy.data.scenes[scene_name]
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    if res_x > res_y:
        sensor_w = camera.data.sensor_width
        sensor_h = camera.data.sensor_width * res_y / res_x
    else:
        sensor_h = camera.data.sensor_width
        sensor_w = camera.data.sensor_width * res_x / res_y

    right_edge = direction + right_vec * sensor_w * 0.5 / camera.data.lens
    left_edge = direction - right_vec * sensor_w * 0.5 / camera.data.lens
    upper_edge = direction + up_vec * sensor_h * 0.5 / camera.data.lens
    lower_edge = direction - up_vec * sensor_h * 0.5 / camera.data.lens

    return (direction, right_edge, left_edge, upper_edge, lower_edge)


def get_ra_dec(vec):
    """Calculate Right Ascension (RA) and Declination (DEC) in radians."""
    vec = vec.normalized()
    dec = math.asin(vec.z)
    ra = math.acos(vec.x / math.cos(dec))
    return (ra + math.pi, dec)


def get_fov(right_edge, left_edge, upper_edge, lower_edge):
    """Calculate centre and size of current Field of View (FOV) in degrees."""
    ra_max = max(get_ra_dec(right_edge)[0], get_ra_dec(left_edge)[0])
    ra_max = math.degrees(ra_max)
    ra_min = min(get_ra_dec(right_edge)[0], get_ra_dec(left_edge)[0])
    ra_min = math.degrees(ra_min)

    if math.fabs(ra_max - ra_min) > math.fabs(ra_max - (ra_min + 360)):
        ra = (ra_min + ra_max + 360) / 2
        if ra >= 360:
            ra -= 360
        width = math.fabs(ra_max - (ra_min + 360))
    else:
        ra = (ra_max + ra_min) / 2
        width = (ra_max - ra_min)

    dec_min = math.degrees(get_ra_dec(lower_edge)[1])
    dec_max = math.degrees(get_ra_dec(upper_edge)[1])
    dec = (dec_max + dec_min) / 2
    height = (dec_max - dec_min)

    return (ra, dec, width, height)


 