"""Main simulation module."""

import math
import subprocess
import sys
import time
import os
import copy
from array import array

import numpy as np
import matplotlib.pyplot as plt
import skimage.filters
import skimage.transform
import simplejson as json
import orekit
from orekit.pyhelpers import setup_orekit_curdir
import org.orekit.orbits as orbits
import org.orekit.utils as utils
from org.orekit.utils import PVCoordinates
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate,TimeScalesFactory
#from org.orekit.data import DataProvidersManager
#from org.orekit.data import DirectoryCrawler
from org.orekit.propagation.analytical import KeplerianPropagator
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.propagation.events import DateDetector
from org.orekit.propagation.events.handlers import RecordAndContinue
from org.orekit.propagation.events.handlers import EventHandler
from org.orekit.python import PythonEventHandler, PythonOrekitFixedStepHandler
from mpl_toolkits.mplot3d import Axes3D
import bpy
#import scipy
#import cv2
#import Imath

import OpenEXR
import blender_controller

#from mathutils import Matrix, Vector, Quaternion, Euler 

#from contextlib import redirect_stdout, redirect_stderr
#import io

orekit.initVM()
setup_orekit_curdir()

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

SERIES_NAME = 'Didymos2OnlyForRec_100kmDepth300kmRotUHSOptLinearDidymoonBetter'
TIME_STEPS = 2#500#1000#1000#50#1000
FACTOR = 10#15#12#10#5#7#Higher values slow down closest encounter phase
MODE = 1
PASS_DURATION = 2. * 60#*2#3600.*24.*30*4*3
TERMINATOR = True#False#True
SUNNYSIDE = False#True#False
CYCLES_SAMPLES = 48#48#48#24#24#8#6#24
EXPOSURE = 1.554

if TERMINATOR:
    SERIES_NAME += '_terminator'
else:
    if SUNNYSIDE:
        SERIES_NAME += '_sunnyside'
    else:
        SERIES_NAME += '_darkside'
SERIES_NAME += str(TIME_STEPS) + '_'

if len(sys.argv) < 2:
    TEMP_PATH = DIR_PATH + '\\temp\\didymos'
else:
    TEMP_PATH = sys.argv[1]

SERIES_PATH = TEMP_PATH + '/' + SERIES_NAME
if not os.path.isdir(SERIES_PATH):
    os.makedirs(SERIES_PATH)

class TimingEvent(PythonEventHandler):
    """TiminigEvent handler."""
    def __init__(self):
        """Initialise a TimingEvent handler."""
        PythonEventHandler.__init__(self)
        self.data = []
        self.events = 0

    def handle_event(self, s, detector, increasing):
        """Handle event occurance."""
        self.events += 1
        if self.events%100 == 0:
            print(s.getDate(), " : event %d"%(self.events))
        
        self.data.append(s)
        return EventHandler.Action.CONTINUE
    
    def reset_state(self, detector, old_state):
        """Reset TimingEvent handler to given state."""
        return old_state

class TimeSampler(DateDetector):
    """."""
    def __init__(self, start, end, steps, mode=1, factor=2):#mode=1 linear, mode=2 double exponential
        """Initialise TimeSampler."""
        duration = end.durationFrom(start)
        dt = duration / (steps-1)
        dtout = dt
        self.times = []
        t = 0.
        self.recorder = RecordAndContinue()
        if mode == 1:
            for i in range(0, steps):
                self.times.append(start.getDate().shiftedBy(t))
                t += dt
        elif mode == 2:
            halfdur = duration / 2.
            
            for i in range(0, steps):
                t2 = halfdur + math.sinh((t-halfdur) * factor/halfdur) * halfdur/math.sinh(factor)
                self.times.append(start.getDate().shiftedBy(t2))
                t += dt
            dtout = duration * math.sinh(factor/steps) / math.sinh(factor)

        print(dtout)
        DateDetector.__init__(self, dtout/2., 1., self.times)


#Didymos data https://ssd.jpl.nasa.gov/horizons.cgi
didymos_a =	1.644641475071416E+00 * utils.Constants.IAU_2012_ASTRONOMICAL_UNIT
didymos_P = 7.703805051391988E+02 * utils.Constants.JULIAN_DAY
didymos_e = 3.838774437558215E-01
didymos_i = math.radians(3.408231185574551E+00)
didymos_omega = math.radians(3.192958853076784E+02)
didymos_Omega = math.radians(7.320940216397703E+01)
didymos_M = math.radians(1.967164895190036E+02)

UTC = TimeScalesFactory.getTDB()
INITIAL_DATE = AbsoluteDate(2017, 8, 19, 0, 0, 0.000, UTC)
INERTIAL_FRAME_EPHEMERIS = FramesFactory.getICRF()
ICRF = FramesFactory.getICRF()
MU_SUN = 1.32712440018E20


DIDYMOS_ORBIT = orbits.KeplerianOrbit(
    didymos_a, didymos_e, didymos_i, didymos_omega, didymos_Omega, didymos_M,
    orbits.PositionAngle.MEAN, INERTIAL_FRAME_EPHEMERIS, INITIAL_DATE, MU_SUN)
DIDYMOS_PROPAGATOR = KeplerianPropagator(DIDYMOS_ORBIT)


POS_SSSB = []
POS_SAT = []

ORBIT_START = AbsoluteDate(2017, 8, 19, 0, 0, 0.000, UTC)

CLOSEST_APPROACH_TIME = AbsoluteDate(2017, 8, 15, 12, 0, 0.000, UTC)

CLOSEST_APPROACH = DIDYMOS_PROPAGATOR.propagate(CLOSEST_APPROACH_TIME.getDate())
didymos_pos = CLOSEST_APPROACH.getPVCoordinates(ICRF).getPosition()
didymos_vel = CLOSEST_APPROACH.getPVCoordinates(ICRF).getVelocity()



dirvec = didymos_pos.normalize()

CLOSEST_APPROACH_DISTANCE = 1E5*3

if not TERMINATOR:
    if not SUNNYSIDE:
        CLOSEST_APPROACH_DISTANCE *= -1
    dirvec = dirvec.scalarMultiply(CLOSEST_APPROACH_DISTANCE)
    sat_pos = didymos_pos.subtract(dirvec)#Minimum distance 1000km approx

else:
    shiftvec = dirvec.scalarMultiply(-0.15)
    shiftvec = shiftvec.add(Vector3D(0., 0., 1.))
    shiftvec = shiftvec.normalize()
    shiftvec = shiftvec.scalarMultiply(CLOSEST_APPROACH_DISTANCE)
    sat_pos = didymos_pos.add(shiftvec)

sat_vel = didymos_vel.scalarMultiply((didymos_vel.getNorm()-10000.) / didymos_vel.getNorm())#0.95)
print("Relative vel", (didymos_vel.subtract(sat_vel)), " len ",
      didymos_vel.subtract(sat_vel).getNorm())
print("Distance from sun", didymos_pos.getNorm() / utils.Constants.IAU_2012_ASTRONOMICAL_UNIT)

sat_orbit = orbits.KeplerianOrbit(PVCoordinates(sat_pos, sat_vel), ICRF, CLOSEST_APPROACH_TIME, MU_SUN)
kepler_sat = KeplerianPropagator(sat_orbit)
kepler_sat_long = KeplerianPropagator(sat_orbit)
min_timestep = PASS_DURATION * math.sinh(FACTOR/TIME_STEPS) / math.sinh(FACTOR)
kepler_long  =  KeplerianPropagator(DIDYMOS_ORBIT)

time_steps_long = 2000
long_orbit_start = CLOSEST_APPROACH_TIME.getDate().shiftedBy(-3600. * 24. * 365 * 2)
long_orbit_end = CLOSEST_APPROACH_TIME.getDate().shiftedBy(3600. * 24. * 365 * 2)

time_sample_handler_long = TimingEvent().of_(TimeSampler)
time_sampler_long = TimeSampler(long_orbit_start, long_orbit_end, time_steps_long,
                                mode=1).withHandler(time_sample_handler_long)
kepler_sat_long.addEventDetector(time_sampler_long)

time_sample_handler2_long = TimingEvent().of_(TimeSampler)
time_sampler2_long = TimeSampler(long_orbit_start, long_orbit_end, time_steps_long,
                                mode=1).withHandler(time_sample_handler2_long)
kepler_long.addEventDetector(time_sampler2_long)

kepler_sat_long.propagate(long_orbit_start.getDate(), long_orbit_end.getDate())
print("Propagating asteroid")
kepler_long.propagate(long_orbit_start.getDate(), long_orbit_end.getDate())


long_orbit_file = open(TEMP_PATH + '/%s/%s_long_orbit.txt'%(SERIES_NAME, SERIES_NAME), 'wt')
for (didymos, sat) in zip(time_sample_handler2_long.data, time_sample_handler_long.data):
    a = didymos    
    b = sat
    pvc = a.getPVCoordinates(ICRF)
    pvc2 = b.getPVCoordinates(ICRF)
    sat_pos = np.asarray(pvc2.getPosition().toArray())
    asteroid_pos = np.asarray(pvc.getPosition().toArray())
    #a.getDate()
    long_orbit_file.write(str(a.getDate()) + ' '
                          + str(asteroid_pos).replace('[', '').replace(']', '') + ','
                          + str(sat_pos).replace('[', '').replace(']', '') + '\n')
long_orbit_file.close()

detector_start = CLOSEST_APPROACH_TIME.getDate().shiftedBy(-PASS_DURATION/2.)
detector_end = CLOSEST_APPROACH_TIME.getDate().shiftedBy(PASS_DURATION/2.)
time_sample_handler = TimingEvent().of_(TimeSampler)
time_sampler = TimeSampler(detector_start, detector_end, TIME_STEPS, MODE, 
                            factor=FACTOR).withHandler(time_sample_handler)
kepler_sat.addEventDetector(time_sampler)

time_sample_handler2 = TimingEvent().of_(TimeSampler)
time_sampler2 = TimeSampler(detector_start, detector_end, TIME_STEPS, MODE, 
                            factor=FACTOR).withHandler(time_sample_handler2)
DIDYMOS_PROPAGATOR.addEventDetector(time_sampler2)



distances = []

print("Starting propagator")
print("Propagating satellite")
kepler_sat.propagate(detector_start.getDate(), detector_end.getDate())
print("Propagating asteroid")
DIDYMOS_PROPAGATOR.propagate(detector_start.getDate(), detector_end.getDate())
print("Propagated")
#print(time_sample_handler.data) 

blender = blender_controller.BlenderController(TEMP_PATH + '/scratch/', 
                                               scene_names=[
                                                   'MainScene',
                                                   'BackgroundStars',
                                                   'AsteroidOnly',
                                                   'AsteroidConstDistance',
                                                   'LightingReference'])
if len(sys.argv) < 3:
    blender.set_renderer('Auto', 128, 512)
else:
    blender.set_renderer(sys.argv[2], 128, 512)

if len(sys.argv) < 6:
    start_frame = 0
    end_frame = TIME_STEPS
    skip_frame = 1
else:
    start_frame = int(sys.argv[3])
    end_frame = int(sys.argv[4])
    skip_frame = int(sys.argv[5])

print("Start %d end %d skip %d"%(start_frame, end_frame, skip_frame))

blender.set_samples(CYCLES_SAMPLES)
blender.set_output_format(2464, 2056)
blender.set_camera(lens=230, sensor=3.45E-3*2464, camera_name='SatelliteCamera',
                    scene_names=['MainScene','BackgroundStars','AsteroidOnly'])
blender.set_camera(lens=230, sensor=3.45E-3*2464, camera_name='ConstantDistanceCamera',
                    scene_names=['AsteroidConstDistance'])
blender.set_camera(lens=230, sensor=3.45E-3*2464, camera_name='LightingReferenceCamera',
                    scene_names=['LightingReference'])

asteroid_scenes = ['MainScene', 'AsteroidOnly', 'AsteroidConstDistance']
star_scenes = ['MainScene', 'BackgroundStars']

Asteroid = blender.load_object(DIR_PATH + "\\Didymos\\didymos2.blend", "Didymos.001",
                                asteroid_scenes)
AsteroidBC = blender.create_empty('AsteroidBC', asteroid_scenes)
MoonOrbiter = blender.create_empty('MoonOrbiter', asteroid_scenes)
Asteroid.parent = AsteroidBC
Asteroid.rotation_mode = 'AXIS_ANGLE'
MoonOrbiter.rotation_mode = 'AXIS_ANGLE'

MoonOrbiter.parent = AsteroidBC
MoonBC = blender.create_empty('MoonBC', asteroid_scenes)
MoonBC.parent = MoonOrbiter
MoonBC.location = (1.17, 0, 0)


Moon = blender.load_object(DIR_PATH + "\\Didymos\\didymos2.blend", "Didymos", asteroid_scenes)
Moon.location = (0, 0, 0)
Moon.parent = MoonBC

Sun = blender.load_object(DIR_PATH + "\\Didymos\\didymos_lowpoly.blend", "Sun",
                            asteroid_scenes + ['LightingReference'])

CalibrationDisk = blender.load_object(DIR_PATH + "\\Didymos\\didymos_lowpoly.blend",
                                        "CalibrationDisk", ['LightingReference'])
CalibrationDisk.location = (0, 0, 0)

frame_index = 0

star_template = blender.load_object(DIR_PATH + "\\Didymos\\StarTemplate.blend", "TemplateStar", 
                                    star_scenes)
star_template.location = (1E20,1E20,1E20)

def get_RA_DEC(vec):
    """Calculate Right Ascension and DEClination."""
    vec = vec.normalized()
    dec = math.asin(vec.z)
    
    ra = math.acos(vec.x / math.cos(dec))
    return (ra + math.pi, dec)

def get_FOV_RA_DEC(leftedge_vec, rightedge_vec, downedge_vec, upedge_vec):
    """Calculate field of view centre and size."""
    ra_max = max(math.degrees(get_RA_DEC(rightedge_vec)[0]), math.degrees(get_RA_DEC(leftedge_vec)[0]))
    ra_min = min(math.degrees(get_RA_DEC(rightedge_vec)[0]), math.degrees(get_RA_DEC(leftedge_vec)[0]))
    
    if(math.fabs(ra_max - ra_min) > math.fabs(ra_max - (ra_min+360))):
        ra_cent = (ra_min+ra_max+360) / 2
        if(ra_cent >= 360):
            ra_cent -= 360
        ra_w = math.fabs(ra_max-(ra_min+360))
    else:
        ra_cent = (ra_max+ra_min) / 2
        ra_w = (ra_max-ra_min)

    dec_min = math.degrees(get_RA_DEC(downedge_vec)[1])
    dec_max = math.degrees(get_RA_DEC(upedge_vec)[1])
    dec_cent = (dec_max+dec_min) / 2
    dec_w = (dec_max-dec_min)


    #print(('RA',ra_cent,'+-',ra_w,'DEC',dec_cent,'+-',dec_w))
    return ra_cent, ra_w, dec_cent, dec_w

errorlog = 'starfield_errorlog%f.txt'%time.time()

def get_UCAC4_data(RA, RA_W, DEC, DEC_W, fn='ucac4.txt'):
    """Retrieve starmap data from UCAC4 catalog."""
    global errorlog
    if sys.platform.startswith("win"):
        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        # How to suppress crash notification dialog?, Jan 14,2004 -
        # Raymond Chen's response [1]

        import ctypes
        SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
        ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
        #subprocess_flags = 0x8000000 #win32con.CREATE_NO_WINDOW?
    #else:
        #subprocess_flags = 0
    #command='%s %f %f %f %f -h %s %s'%(ucac_exe,self.ra_cent,self.dec_cent,self.ra_w,self.dec_w,ucac_data,ucac_out)
    
    command = 'E:\\01_MasterThesis\\00_Code\\star_cats\\u4test.exe %f %f %f %f -h E:\\01_MasterThesis\\02_Data\\UCAC4 %s'%(RA, DEC, RA_W, DEC_W, fn)
    print(command)
    
    for i in range(0, 5):
        retcode = subprocess.call(command)
        print("Retcode ", retcode)
        if retcode == 0:
            break
        fout = open(errorlog,'at')
        fout.write("%f,\'%s\',%d\n"%(time.time(), command, retcode))
        fout.close()
    file = open(fn, 'rt')
    lines = file.readlines()
    print('Lines',len(lines))
    out = []
    for line in lines[1:]:

        r = float(line[11:23])
        d = float(line[23:36])
        m = float(line[36:43])
        #print((r,d,m))
        out.append([r, d, m])
    return out

def write_openEXR(fn, picture):
    """Save image in OpenEXR file format."""
    h = len(picture)
    w = len(picture[0])
    c = len(picture[0][0])

    hdr = OpenEXR.Header(w, h)
    x = OpenEXR.OutputFile(fn, hdr)

    if c == 4:
        dataR = picture[:,:,0].tobytes()
        dataG = picture[:,:,1].tobytes()
        dataB = picture[:,:,2].tobytes()
        dataA = picture[:,:,3].tobytes()
        x.writePixels({'R': dataR, 'G': dataG, 'B': dataB,'A':dataA})
    elif c == 3:
        dataR = picture[:,:,0].tobytes()
        dataG = picture[:,:,1].tobytes()
        dataB = picture[:,:,2].tobytes()

        x.writePixels({'R': dataR, 'G': dataG, 'B': dataB})
    x.close()

class StarCache:
    """Handling stars in field of view, for rendering of scene."""
    def __init__(self, template, parent=None):
        self.template = template
        self.star_array = []
        self.parent = parent

    def set_stars(self, stardata, cam_direction, sat_position, R, pixelsize_at_R, scene_names):
        """Set current stars in the field of view."""
        if len(self.star_array) < len(stardata):
            for i in range(0, len(stardata) - len(self.star_array)):
                new_obj = self.template.copy()
                new_obj.data = self.template.data.copy()
                new_obj.animation_data_clear()
                new_mat = star_template.material_slots[0].material.copy()
                new_obj.material_slots[0].material = new_mat
                self.star_array.append(new_obj)
                if self.parent != None:
                    new_obj.parent = self.parent
        total_flux = 0.

        for i in range(0, len(stardata)):
            star = self.star_array[i]
            star_data = copy.copy(stardata[i])
            star_data[0] = math.radians(star_data[0])
            star_data[1] = math.radians(star_data[1])

            z = math.sin(star_data[1])
            x = math.cos(star_data[1])*math.cos(star_data[0]-math.pi)
            y = -math.cos(star_data[1])*math.sin(star_data[0]-math.pi)
            vec = [x, y, z]
            vec2 = [x, -y, z]
            if(np.dot(vec,cam_direction)<np.dot(vec2,cam_direction)):
                vec = vec2
            
            pixel_factor = 10
            star.location = np.asarray(vec)*R + sat_pos_rel#Always keep satellite in center to emulate large distances
            star.scale = (pixelsize_at_R/pixel_factor, pixelsize_at_R/pixel_factor, 
                            pixelsize_at_R/pixel_factor)

            flux = math.pow(10, -0.4 * (star_data[2]-10.)) 
            flux0 = math.pow(10, -0.4 * (star_data[2])) 
            total_flux += flux0

            star.material_slots[0].material.node_tree.nodes.get('Emission').inputs[1].default_value = flux * pixel_factor * pixel_factor
            
            for scene_name in scene_names:
                scene = bpy.data.scenes[scene_name]
                if star.name not in scene.objects:
                    scene.objects.link(star)
        print("%d stars set, buffer len %d"%(i, len(self.star_array)))
        if len(self.star_array) > len(stardata):
            for scene_name in scene_names:
                scene = bpy.data.scenes[scene_name]
                for i in range(len(stardata), len(self.star_array)):
                    if self.star_array[i].name in scene.objects:
                        scene.objects.unlink(self.star_array[i])

        return total_flux

    def render_stars_directly(self, stardata, cam_direction, right_vec, up_vec, res_x, res_y, fn):
        """Render given stars."""
        up_vec -= cam_direction
        right_vec -= cam_direction
        total_flux = 0.

        f_over_h_ccd_2 = 1. / np.sqrt(np.dot(up_vec, up_vec))
        up_norm = up_vec * f_over_h_ccd_2
        f_over_w_ccd_2 = 1. / np.sqrt(np.dot(right_vec, right_vec))
        right_norm = right_vec * f_over_w_ccd_2

        print('F_over_w %f f_over_h %f'%(f_over_w_ccd_2, f_over_h_ccd_2))
        print('Res %d x %d'%(res_x,res_y))

        ss = 2
        #rx = res_x * ss
        #ry = res_y * ss
        starmap = np.zeros((res_y * ss, res_x * ss, 4), np.float32)

        for i in range(0,len(stardata)):
            
            star_data = copy.copy(stardata[i])

            star_data[0] = math.radians(star_data[0])
            star_data[1] = math.radians(star_data[1])

            z = math.sin(star_data[1])
            x = math.cos(star_data[1]) * math.cos(star_data[0] - math.pi)
            y = -math.cos(star_data[1]) * math.sin(star_data[0] - math.pi)
            vec = [x, y, z]
            vec2 = [x, -y, z]
            if(np.dot(vec, cam_direction) < np.dot(vec2, cam_direction)):
                vec = vec2
            
            x_pix = ( f_over_w_ccd_2 * np.dot(right_norm,vec) / np.dot(cam_direction,vec)+1.) * (res_x-1)/2.
            y_pix = (-f_over_h_ccd_2 * np.dot(up_norm,vec) /np.dot(cam_direction,vec)+1.) * (res_y-1)/2.
            x_pix2 = max(0, min(int(round(x_pix*ss)), res_x*ss-1))
            y_pix2 = max(0, min(int(round(y_pix*ss)), res_y*ss-1))

            flux = math.pow(10.,-0.4*(star_data[2])) 
            flux0 = math.pow(10.,-0.4*(star_data[2])) 
            
            pix = starmap[y_pix2, x_pix2]
            starmap[y_pix2, x_pix2] = [pix[0]+flux, pix[1]+flux, pix[2]+flux, 1.]

            total_flux += flux0
        starmap2 = starmap.copy()
        starmap2 = skimage.filters.gaussian(starmap,ss/2.,multichannel = True)
        starmap3 = np.zeros((res_y,res_x,4), np.float32)
        for c in range(0,4):
            
            starmap3[:,:,c] = skimage.transform.downscale_local_mean(starmap2[:,:,c], (ss,ss)) * (ss*ss)

        #starmap2 = skimage.transform.rescale(starmap2,1./ss,mode = 'constant')#,multichannel = True)
        #for c in range(0,3):
        #    starmap2[:,:,c]*=flux*(1E4)/np.sum(starmap2[:,:,c])
        #starmap2 = np.asarray(starmap2,dtype = 'float32')
        write_openEXR(fn, starmap3)

        return (total_flux, np.sum(starmap3[:,:,0]))
 
def vec_string(vec, prec):
    """Create string from vector data."""
    o = '['
    fs = '%%.%de'%(prec)
    #i = 0
    for (n,v) in enumerate(vec):

        o += fs%(v)
        if n < len(vec) - 1:
            o += ','
    return o + ']'

def mat_string(vec, prec):
    """Create string from matrix data."""
    o = '['
    #fs = '%%.%de'%(prec)
    i = 0
    for (n,v) in enumerate(vec):

        o += (vec_string(v,prec))
        if n < len(vec) - 1:
            o += ','
    return o + ']'


star_cache = StarCache(star_template, blender.create_empty("StarParent", star_scenes))
#cmd = 'mkdir "'+scratchloc+'/'+series_name+'"'
#print(cmd)

#subprocess.call(cmd)
ucac_fn = TEMP_PATH + '/%s/ucac4_%d.txt'%(SERIES_NAME, time.time())
scaler = 1000.
blender.set_exposure(EXPOSURE)
for (didymos, sat, frame_index) in zip(time_sample_handler2.data[start_frame:end_frame:skip_frame],
                                        time_sample_handler.data[start_frame:end_frame:skip_frame],
                                        range(0,TIME_STEPS)[start_frame:end_frame:skip_frame]):
    #if frame_index<332:
   #     continue
    
    a = didymos
    t = a.getDate().durationFrom(detector_start)
    halfdur = detector_end.durationFrom(detector_start) / 2
    print("Starting frame %d time = %f"%(frame_index,t-halfdur))

    b = sat
    pvc = a.getPVCoordinates(ICRF)
    pvc2 = b.getPVCoordinates(ICRF)
    distance = Vector3D.distance(pvc.getPosition(), pvc2.getPosition())

    sat_pos = np.asarray(pvc2.getPosition().toArray())
    asteroid_pos = np.asarray(pvc.getPosition().toArray())

    sat_pos_rel = (sat_pos-asteroid_pos) / scaler

    satellite_camera = blender.cameras['SatelliteCamera']
    satellite_camera.location = sat_pos_rel

    constant_distance_camera = blender.cameras['ConstantDistanceCamera']
    constant_distance_camera.location = sat_pos_rel*1E3 / np.sqrt(np.dot(sat_pos_rel,sat_pos_rel))

    reference_camera = blender.cameras['LightingReferenceCamera']
    reference_camera.location = -asteroid_pos*1E3/np.sqrt(np.dot(asteroid_pos,asteroid_pos))

    Sun.location = -asteroid_pos/scaler
    
    asteroid_rotation = 2.*math.pi*t / (2.2593*3600)
    Asteroid.rotation_axis_angle = (asteroid_rotation, 0, 0, 1)

    moon_orbiter = 2.*math.pi*t / (11.9*3600)
    MoonOrbiter.rotation_axis_angle = (moon_orbiter, 0, 0, 1)

    blender.target_camera(Asteroid, 'SatelliteCamera')
    blender.target_camera(Asteroid, 'ConstantDistanceCamera')
    blender.target_camera(Sun, "CalibrationDisk")#A bit unorthodox use
    blender.target_camera(CalibrationDisk, 'LightingReferenceCamera')
    blender.update()

    (cam_direction, up, right, leftedge_vec, rightedge_vec, downedge_vec, upedge_vec) = blender.get_camera_vectors('SatelliteCamera', 'MainScene')

    (ra_cent, ra_w, dec_cent, dec_w) = get_FOV_RA_DEC(leftedge_vec, rightedge_vec, downedge_vec, 
                                                        upedge_vec)
    
    starlist = get_UCAC4_data(ra_cent, ra_w,dec_cent, dec_w, ucac_fn)


    #R = 100000000.
    #pixelsize_at_R = R*math.radians(ra_w)/blender.scenes['BackgroundStars'].render.resolution_x
    #print("Pixel size at %f is %f"%(R,pixelsize_at_R))
    #i = 0
    print("Found %d stars in FOV"%(len(starlist)))
    #starfield_flux = star_cache.SetStars(starlist,cam_direction,sat_pos_rel,R,pixelsize_at_R,star_scenes)


    x_res = blender.scenes['BackgroundStars'].render.resolution_x
    y_res = blender.scenes['BackgroundStars'].render.resolution_y    
    f = blender.cameras['SatelliteCamera'].data.lens
    w = blender.cameras['SatelliteCamera'].data.sensor_width

    fn_base5 = TEMP_PATH + '/%s/%s_starmap_direct_%.4d.exr'%(SERIES_NAME, SERIES_NAME,
                                                                frame_index)
    (starfield_flux2, flux3) = star_cache.render_stars_directly(starlist, cam_direction,
                                                                rightedge_vec, 
                                                                upedge_vec, x_res, y_res, fn_base5)

    blender.update()
    fn_base = TEMP_PATH + '/%s/%s%.4d'%(SERIES_NAME, SERIES_NAME, frame_index)
    print("Saving blend file")
    #bpy.ops.wm.save_as_mainfile(filepath = fn_base+'.blend')
    print("Rendering")
    
    
    #result = blender.Render(fn_base,'MainScene')

    #fn_base2 = scratchloc+'/%s/%s_stars_%.4d'%(series_name,series_name,frame_index)
    #result = blender.Render(fn_base2,'BackgroundStars')

    fn_base3 = TEMP_PATH + '/%s/%s_asteroid_%.4d'%(SERIES_NAME, SERIES_NAME, frame_index)
    blender.update(['AsteroidOnly'])
    
    result = blender.render(fn_base3, 'AsteroidOnly')


    fn_base4 = TEMP_PATH + '/%s/%s_asteroid_constant_%.4d'%(SERIES_NAME, SERIES_NAME, frame_index)
    blender.update(['AsteroidConstDistance'])
    
    result = blender.render(fn_base4, 'AsteroidConstDistance')

    fn_base6 = TEMP_PATH+'/%s/%s_calibration_reference_%.4d'%(SERIES_NAME, SERIES_NAME, frame_index)
    blender.update(['LightingReference'])
    result = blender.render(fn_base6, 'LightingReference')

    print("Rendering complete")
    bpy.ops.wm.save_as_mainfile(filepath=fn_base+'.blend')
    
    metafile = open(fn_base+'.txt', 'wt')



    metafile.write('%s time\n'%(a.getDate()))
    metafile.write('%s distance (m)\n'%(distance))
    metafile.write('%e %e total_flux (in Mag 0 units)\n'%(starfield_flux2, flux3))

    metafile.write('%s Didymos (m)\n'%(vec_string(asteroid_pos, 17)))
    metafile.write('%s Satellite (m)\n'%(vec_string(sat_pos, 17)))
    metafile.write('%s Satellite relative \n'%(vec_string(sat_pos_rel, 17)))
    metafile.write('%s Satellite matrix \n'%(mat_string(satellite_camera.matrix_world, 17)))
    metafile.write('%s Asteroid matrix \n'%(mat_string(Asteroid.matrix_world, 17)))
    metafile.write('%s Sun matrix \n'%(mat_string(Sun.matrix_world, 17)))
    metafile.write('%s Constant distance matrix \n'%(mat_string(constant_distance_camera.matrix_world, 17)))
    metafile.write('%s Reference matrix \n'%(mat_string(reference_camera.matrix_world, 17)))
    metafile.write('%s Camera f,w,x,y \n'%(vec_string([f, w, x_res, y_res], 17)))

    metafile.close()
    
    metadict = dict()
    metadict['time'] = a.getDate()
    metadict['time_t'] = t
    metadict['distance (m)'] = distance
    metadict['total_flux (in Mag 0 units)'] = (starfield_flux2, flux3)

    metadict['Didymos (m)'] = asteroid_pos
    metadict['Satellite (m)\n'] = sat_pos
    metadict['Satellite relative'] = sat_pos_rel
    metadict['Satellite matrix'] = satellite_camera.matrix_world
    metadict['Asteroid matrix '] = Asteroid.matrix_world
    metadict['Sun matrix'] = Sun.matrix_world
    metadict['Constant distance matrix'] = constant_distance_camera.matrix_world
    metadict['Reference matrix'] = reference_camera.matrix_world
    metadict['Camera f,w,x,y'] = (f,w,x_res,y_res)

    def serializer(o):
        """."""
        try:
            return np.asarray(o, dtype='float64').tolist()
        except:
            try:
                return float(o)
            except:
                return str(o)

    with open(fn_base+'.json', 'w') as _file:
        json.dump(metadict, _file, default=serializer)

    distances.append([t, distance])
    POS_SSSB.append(asteroid_pos)
    POS_SAT.append(sat_pos)


    print("Frame %d complete"%(frame_index))
    #frame_index+ = 1
    #break

  
fig = plt.figure(1)
POS_SSSB = np.asarray(POS_SSSB, dtype='float64').transpose()
POS_SAT = np.asarray(POS_SAT, dtype='float64').transpose()
distances = np.asarray(distances, dtype='float64').transpose()
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot(POS_SSSB[0], POS_SSSB[1], POS_SSSB[2])
ax.plot(POS_SAT[0], POS_SAT[1], POS_SAT[2])

au=utils.Constants.IAU_2012_ASTRONOMICAL_UNIT
#ax.set_xlim(-3*au,3*au)
#ax.set_ylim(-3*au,3*au)
#ax.set_zlim(-3*au,3*au)
plt.figure(2)
plt.clf()
plt.plot(distances[0], distances[1])

plt.show()
