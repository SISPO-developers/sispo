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

import utils

logger = utils.create_logger("rendering")


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

        # Set scene name to MainScene and clear everything
        bpy.context.scene.name = "MainScene"
        for obj in bpy.data.objects:
            obj.select_set(True)
        bpy.ops.object.delete()

        self.set_device()

        self.render_id = zlib.crc32(struct.pack("!f", time.time()))

    def create_scene(self, scene_name):
        """Add empty scene."""
        bpy.ops.scene.new(type="FULL_COPY")
        bpy.context.scene.name = scene_name

        self.set_scene_defaults([scene_name])

        bpy.context.window.scene = self.default_scene

    def set_scene_defaults(self, scenes=None):
        """Sets default settings to a scene."""
        for scene in self._get_scenes_iter(scenes):
            scene.render.image_settings.color_mode = "RGBA"
            scene.render.image_settings.use_zbuffer = True
            scene.render.resolution_percentage = 100 # TODO: why 100? int in [1, 32767], default 0
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

    def set_resolution(self, res_x, res_y, scenes=None):
        """Sets resolution of rendered image."""
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

    def set_output_file(self, filename, scene_name):
        """Set output file path to given scenes with prior extension check."""
        filename = str(filename)

        file_extension = ".exr"
        if filename[-4:] != file_extension:
            filename += file_extension

        bpy.data.scenes[scene_name].render.filepath = str(filename)

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

    def render(self, name=None, scene_name="MainScene"):
        """Render given scene."""
        if name is None:
            name = self.res_dir / f"r{self.render_id:0.8X}"
        
        self.set_output_file(name, scene_name)

        bpy.context.window.scene = bpy.data.scenes[scene_name]
        bpy.ops.render.render(write_still=True)

    def load_object(self, filename, object_name, scenes=None):
        """Load blender object from file."""
        filename = str(filename)

        with bpy.data.libraries.load(filename) as (data_from, data_to):
            data_to.objects = [
                name for name in data_from.objects if name == object_name]
        if data_to.objects:
            obj = data_to.objects[0]
            obj.animation_data_clear()
            
            for scene_name in self._get_scenes_iter(scenes):
                bpy.data.scenes[scene_name].collection.objects.link(obj)
            return obj
        else:
            msg = f"{object_name} not found in {filename}"
            logger.info(msg)
            raise BlenderControllerError(msg)

    def create_empty(self, name="Empty", scenes=None):
        """Create new, empty blender object."""
        obj_empty = bpy.data.objects.new(name, None)
        for scene_name in self._get_scenes_iter(scenes):
            scene = bpy.data.scenes[scene_name]
            scene.collection.objects.link(obj_empty)
        return obj_empty

    def save_blender_dfile(self, filename):
        """Save a blender d file."""
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
            output = self.scenes[scenes]
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


def get_camera_vectors(camera_name, scene_name):
    """Get camera position and direction vectors."""
    camera = bpy.data.objects[camera_name]
    up_vec = camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
    cam_direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    right_vec = cam_direction.cross(up_vec)

    scene = bpy.data.scenes[scene_name]
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    if res_x > res_y:
        sensor_w = camera.data.sensor_width
        sensor_h = camera.data.sensor_width * res_y / res_x
    else:
        sensor_h = camera.data.sensor_width
        sensor_w = camera.data.sensor_width * res_x / res_y

    rightedge_vec = cam_direction + right_vec * sensor_w * 0.5 / camera.data.lens
    leftedge_vec = cam_direction - right_vec * sensor_w * 0.5 / camera.data.lens
    upedge_vec = cam_direction + up_vec * sensor_h * 0.5 / camera.data.lens
    downedge_vec = cam_direction - up_vec * sensor_h * 0.5 / camera.data.lens

    return cam_direction, up_vec, right_vec, leftedge_vec, rightedge_vec, downedge_vec, upedge_vec


def get_ra_dec(vec):
    """Calculate Right Ascension (RA) and Declination (DEC) in radians."""
    vec = vec.normalized()
    dec = math.asin(vec.z)
    ra = math.acos(vec.x / math.cos(dec))
    return (ra + math.pi, dec)


def get_fov(leftedge_vec, rightedge_vec, downedge_vec, upedge_vec):
    """Calculate centre and size of a camera's current Field of View (FOV) in degrees."""
    ra_max = max(get_ra_dec(rightedge_vec)[0], get_ra_dec(leftedge_vec)[0])
    ra_max = math.degrees(ra_max)
    ra_min = min(get_ra_dec(rightedge_vec)[0], get_ra_dec(leftedge_vec)[0])
    ra_min = math.degrees(ra_min)

    if math.fabs(ra_max - ra_min) > math.fabs(ra_max - (ra_min + 360)):
        ra_cent = (ra_min + ra_max + 360) / 2
        if ra_cent >= 360:
            ra_cent -= 360
        ra_w = math.fabs(ra_max - (ra_min + 360))
    else:
        ra_cent = (ra_max + ra_min) / 2
        ra_w = (ra_max - ra_min)

    dec_min = math.degrees(get_ra_dec(downedge_vec)[1])
    dec_max = math.degrees(get_ra_dec(upedge_vec)[1])
    dec_cent = (dec_max + dec_min) / 2
    dec_w = (dec_max - dec_min)

    return ra_cent, ra_w, dec_cent, dec_w
 