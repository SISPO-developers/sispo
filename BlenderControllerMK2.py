import time
import zlib
import struct



import bpy
from mathutils import Vector


class BlenderController:
    def __init__(self, scratchdisk, scene_names = ['MainScene']):
        self.scene_names = scene_names
        self.scene = scene = bpy.context.scene
        self.scene.name = scene_names[0]
        self.cameras = bpy.data.cameras
        scene.world.color = (0, 0, 0)
        #Clear everything on the scene
        for ob in bpy.data.objects:
            ob.select_set(True)
        bpy.ops.object.delete()

        if len(scene_names) > 1:
            for scene_name in scene_names[1:]:
                bpy.ops.scene.new(type='FULL_COPY')
                scene = bpy.context.scene
                scene.name = scene_name
        self.scenes = bpy.data.scenes
       
        self.scratchdisk = scratchdisk
        
        self.render_ID = zlib.crc32(struct.pack("!f", time.time()))

    def SetRenderer(self, device = 'Auto', tile = 64, tile_GPU = 512, scene_names = []):
        print("Render setting %r" % (device))
        if len(scene_names) == 0:
            scene_names = self.scene_names

        if device != 'CPU' and device != 'GPU':
            if(len(bpy.context.preferences.addons['cycles'].preferences.devices) > 0):
                device = 'GPU'
                tile = tile_GPU
            else:
                device = 'CPU'

        if device == 'GPU':
            bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            print('Rendering with GPUs:')
            for gpu in bpy.context.user_preferences.addons['cycles'].preferences.devices:
                gpu.use = True
                print(gpu.name)
            tile = tile_GPU
        else:
            print('Rendering with CPUs')

        for scene_name in self.scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.render.engine = 'CYCLES'
            cycles = scene.cycles
            cycles.feature_set = 'EXPERIMENTAL'

            scene.render.resolution_percentage = 100
            cycles.device = device

            scene.render.tile_x = tile
            scene.render.tile_y = tile
            cycles.max_bounces = 128
            cycles.min_bounces = 3
            cycles.caustics_reflective = True
            cycles.caustics_refractive = True
            cycles.diffuse_bounces = 128
            cycles.glossy_bounces = 128
            cycles.transmission_bounces = 128
            cycles.volume_bounces = 128
            cycles.transparent_min_bounces = 8
            cycles.transparent_max_bounces = 128
            cycles.use_square_samples = True
            #cycles.use_animated_seed=True
            cycles.seed = time.time()
            cycles.film_transparent = True
            scene.view_settings.view_transform = 'Raw'
            scene.view_settings.look = 'None'

    def SetExposure(self, exposure):
        for scene_name in self.scene_names:
            scene = bpy.data.scenes[scene_name]
            
            scene.view_settings.exposure = exposure

    def SetOutputFormat(self, res_x, res_y, file_format = 'OPEN_EXR', color_depth = '32', use_preview = True, scene_names = []):
        if len(scene_names) == 0:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.render.image_settings.file_format = file_format
            scene.render.filepath = self.scratchdisk + 'r%0.8X.exr' % (self.render_ID)
            scene.render.resolution_x = res_x
            scene.render.resolution_y = res_y
            scene.render.resolution_percentage = 100
            scene.render.image_settings.color_depth = color_depth
            scene.render.image_settings.color_mode = 'RGBA'
            scene.render.image_settings.use_preview = use_preview
            scene.render.image_settings.use_zbuffer = True

    def SetSamples(self, samples = 6, scene_names = []):
        if len(scene_names) == 0:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.cycles.samples = samples

    def Update(self, scene_names = []):
        if len(scene_names) == 0:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            bpy.context.window.scene = scene
            scene.cycles.seed = time.time()
            scene.update()
    def Render(self, name = '', scene_name = 'MainScene'):
        if name == '':
            name = self.scratchdisk + 'r%0.8X.exr' % (self.render_ID)
              
        scene = bpy.data.scenes[scene_name]
        print("Rendering seed: %d" % (scene.cycles.seed))    
        scene.render.filepath = name
        bpy.context.window.scene = scene
        bpy.ops.render.render(write_still=True)
        # get viewer pixels

        #return cv2.imread(self.scene.render.filepath)

    def LoadObject(self, filename, object_name, scene_names = []):

        with bpy.data.libraries.load(filename) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects if name == object_name]
        if len(data_to.objects) > 0:
            object = data_to.objects[0]
            object.animation_data_clear()
            if len(scene_names) == 0:
                scene_names = self.scene_names
            for scene_name in scene_names:
                scene = bpy.data.scenes[scene_name]
                scene.collection.objects.link(object)
            return object
        return None

    def SetCamera(self, lens = 35, sensor = 32, clip_start = 1E-5, clip_end = 1E32, mode = 'PERSP', ortho_scale = 7, camera_name = 'Camera', scene_names = []):#Modes are 'ORTHO' and 'PERSP'
        cam = bpy.data.cameras.new(camera_name)
        camera = bpy.data.objects.new('Camera', cam)
        camera.name = camera_name
        camera.data.clip_end = clip_end
        camera.data.clip_start = clip_start
        camera.data.lens = lens
        camera.data.ortho_scale = ortho_scale
        camera.data.sensor_width = sensor
        camera.location = (0, 0, 0)
        camera.data.type = mode
        self.cameras = bpy.data.objects
        if len(scene_names) == 0:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.camera = camera
            scene.collection.objects.link(camera)

    def TargetCamera(self, target, camera_name = 'Camera'):
        camera = bpy.data.objects[camera_name]
        camera_constr = camera.constraints.new(type='TRACK_TO')
        camera_constr.track_axis = 'TRACK_NEGATIVE_Z'
        camera_constr.up_axis = 'UP_Y'
        camera_constr.target = target

    def CreateEmpty(self, name = 'Empty', scene_names = []):
        e = bpy.data.objects.new(name, None)
        if len(scene_names) == 0:
            scene_names = self.scene_names
        for scene_name in scene_names:
            scene = bpy.data.scenes[scene_name]
            scene.collection.objects.link(e)
        return e

    def SaveBlenderDfile(self, fn):
        bpy.ops.wm.save_as_mainfile(fn)

    def GetCameraVectors(self, camera_name, scene_name):
        camera = bpy.data.objects[camera_name]
        up = camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
        cam_direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        right = cam_direction.cross(up)

        scene = bpy.data.scenes[scene_name]
        res_x = scene.render.resolution_x
        res_y = scene.render.resolution_y

        #max_dim = max(res_x, res_y)
        if res_x > res_y:
            sensor_w = camera.data.sensor_width
            sensor_h = camera.data.sensor_width * res_y / res_x
        else:
            sensor_h = camera.data.sensor_width
            sensor_w = camera.data.sensor_width * res_x / res_y

        rightedge_vec = cam_direction + right * sensor_w * 0.5 / camera.data.lens
        leftedge_vec = cam_direction - right * sensor_w * 0.5 / camera.data.lens
        upedge_vec = cam_direction + up * sensor_h * 0.5 / camera.data.lens
        downedge_vec = cam_direction - up * sensor_h * 0.5 / camera.data.lens

        return cam_direction, up, right, leftedge_vec, rightedge_vec, downedge_vec, upedge_vec
