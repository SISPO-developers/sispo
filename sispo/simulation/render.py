"""Module to controll blender python module bpy."""

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

    def __init__(self, render_dir, scene_names=None):
        """Initialise blender controller class."""

        self.res_path = render_dir
        self.cycles = bpy.context.preferences.addons["cycles"]

        if scene_names is None:
            scene_names = ["MainScene"]

        self.scene_names = scene_names
        self.scene = scene = bpy.context.scene
        self.scene.name = scene_names[0]
        self.cameras = bpy.data.cameras
        scene.world.color = (0, 0, 0)

        # Clear everything on the scene
        for obj in bpy.data.objects:
            obj.select_set(True)
        bpy.ops.object.delete()

        if len(scene_names) > 1:
            for scene_name in scene_names[1:]:
                bpy.ops.scene.new(type="FULL_COPY")
                scene = bpy.context.scene
                scene.name = scene_name

        self.set_scene_defaults()
        self.set_device()

        self.render_id = zlib.crc32(struct.pack("!f", time.time()))

    def set_scene_defaults(self, scene_names=None):
        """Sets default settings to a scene."""
        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
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

    def set_device(self, device="AUTO", scene_names=None):
        """Set cycles rendering device for given scenes.

        When device="AUTO" it is attempted to use GPU first, otherwise
        fallback is CPU. Currently, assumes set_device is only used once.
        """
        logger.info("Attempting to set cycle rendering device to: %s", device)

        self.device = self._determine_device(device)
        self._set_cycles_device()
        tile_size = self.get_tile_size()

        # Sets render device of scenes
        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
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

    def set_samples(self, samples=6, scene_names=None):
        """Set number of samples to render for each pixel."""
        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in scene_names:
            bpy.data.scenes[scene_name].cycles.samples = samples

    def set_exposure(self, exposure, scene_names=None):
        """Set exposure value."""
        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in self.scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.view_settings.exposure = exposure

    def set_resolution(self, res_x, res_y, scene_names=None):
        """Sets resolution of rendered image."""
        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.render.resolution_x = res_x
            scene.render.resolution_y = res_y

    def set_output_format(self,
                          file_format="OPEN_EXR",
                          color_depth="32",
                          use_preview=True,
                          scene_names=None):
        """Set output file format."""
        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.render.image_settings.file_format = file_format
            scene.render.image_settings.color_depth = color_depth
            scene.render.image_settings.use_preview = use_preview

    def set_camera(self,
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

    def create_camera(self, camera_name="Camera", scene_names=None):
        """Create new camera and add to relevant scenes."""
        cam = bpy.data.cameras.new(camera_name)
        camera = bpy.data.objects.new(camera_name, object_data=cam)
        camera.name = camera_name
        camera.location = (0, 0, 0)

        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.camera = camera
            scene.collection.objects.link(camera)

    def update(self, scene_names=None):
        """Update scenes."""
        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            bpy.context.window.scene = scene
            scene.cycles.seed = time.time()
            scene.view_layers.update()

    def render(self, name=None, scene_name="MainScene"):
        """Render scenes."""
        if name is None:
            name = self.res_path / f"r{self.render_id:0.8X}.exr"

        scene = bpy.data.scenes[scene_name]
        print("Rendering seed: %d" % (scene.cycles.seed))
        scene.render.filepath = str(name)
        bpy.context.window.scene = scene
        bpy.ops.render.render(write_still=True)

    def load_object(self, filename, object_name, scene_names=None):
        """Load blender object from file."""
        filename = str(filename)
        with bpy.data.libraries.load(filename) as (data_from, data_to):
            data_to.objects = [
                name for name in data_from.objects if name == object_name]
        if data_to.objects:
            obj = data_to.objects[0]
            obj.animation_data_clear()
            if scene_names is None:
                scene_names = self.scene_names
            for scene_name in scene_names:
                scene = bpy.data.scenes[scene_name]
                scene.collection.objects.link(obj)
            return obj
        else:
            msg = f"{object_name} not found in {filename}"
            logger.info(msg)
            raise BlenderControllerError(msg)

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

    def create_empty(self, name="Empty", scene_names=None):
        """Create new, empty blender object."""
        obj_empty = bpy.data.objects.new(name, None)
        if scene_names is None:
            scene_names = self.scene_names
        for scene_name in scene_names:
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
 