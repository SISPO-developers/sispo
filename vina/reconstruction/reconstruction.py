#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# This file is part of OpenMVG (Open Multiple View Geometry) C++ library.

# Python script to launch OpenMVG SfM tools on an image dataset
#
# usage : python tutorial_demo.py
#
from os import system
import os
from pathlib import Path

SOFTWARE_DIR = Path.cwd().joinpath("software")


system("title "+'"%s"'%(os.path.abspath(__file__)))
OPENMVG_SFM_BIN = SOFTWARE_DIR.joinpath("openMVG").joinpath("build").joinpath("Windows-AMD64-").joinpath("Release")
OPENMVS_BIN = SOFTWARE_DIR.joinpath("openMVS").joinpath("build").joinpath("bin").joinpath("x64").joinpath("Debug")
# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = SOFTWARE_DIR.joinpath("openMVG").joinpath("openMVG").joinpath("src").joinpath("openMVG").joinpath("exif").joinpath("sensor_width_database")

import subprocess
import sys

root = Path.cwd()
workdir=root.joinpath("data").joinpath("results")
if not workdir.exists():
  Path.mkdir(workdir)
  
#os.chdir(workdir)
input_eval_dir = Path.cwd().joinpath("data").joinpath("ImageDataset_SceauxCastle-master")
# Checkout an OpenMVG image dataset with Git

output_eval_dir = input_eval_dir.joinpath("reconstruction")
#output_eval_dir = os.path.join(get_parent_dir(input_eval_dir), "asteroid_out")

input_eval_dir = input_eval_dir.joinpath("images")
if not input_eval_dir.exists():
  Path.mkdir(input_eval_dir)

if not output_eval_dir.exists():
  Path.mkdir(output_eval_dir)

input_dir = input_eval_dir
output_dir = output_eval_dir
print ("Using input dir  : ", input_dir)
print ("      output_dir : ", output_dir)

matches_dir = output_dir.joinpath("matches")
camera_file_params = CAMERA_SENSOR_WIDTH_DIRECTORY.joinpath("sensor_width_camera_database.txt")

# Create the ouput\\matches folder if not present
if not matches_dir.exists():
  Path.mkdir(matches_dir)

fl=65437    
reconstruction_dir = output_dir.joinpath("reconstruction_sequential")
if not reconstruction_dir.exists():
  Path.mkdir(reconstruction_dir)

reconstruction_dir_scene = reconstruction_dir.joinpath("scene")
if not reconstruction_dir_scene.exists():
  Path.mkdir(reconstruction_dir_scene)

reconstruction_dir2 = output_dir.joinpath("reconstruction_sequential2")
if not reconstruction_dir2.exists():
  Path.mkdir(reconstruction_dir2)

reconstruction_dir_scene2 = reconstruction_dir2.joinpath("scene")
if not reconstruction_dir_scene2.exists():
  Path.mkdir(reconstruction_dir_scene2)

#if 1:    
#  print ("1. Intrinsics analysis")
#  pIntrisics = subprocess.Popen( [str(OPENMVG_SGM_BIN.joinpath("openMVG_main_SfMInit_ImageListing")),  "-i", input_dir, "-o", str(matches_dir), "-d", str(camera_file_params), "-c", "1","-f",str(fl),"-P","-W","1.0;1.0;1.0;"] )
#  pIntrisics.wait()
  
#  print ("2. Compute features")
#  pFeatures = subprocess.Popen( [ str(OPENMVG_SFM_BIN.joinpath("openMVG_main_ComputeFeatures")),  "-i", str(matches_dir.joinpath("sfm_data.json")), "-o", str(matches_dir), "-m", "SIFT", "-f" , "0","-p","ULTRA"] )
#  pFeatures.wait()
#if 1:  
#  print ("2. Compute matches")
#  pMatches = subprocess.Popen( [ str(OPENMVG_SFM_BIN.joinpath("openMVG_main_ComputeMatches")),  "-i", str(matches_dir.joinpath("sfm_data.json")), "-o", matches_dir, "-f", "1", "-n", "FASTCASCADEHASHINGL2","-v","12"] )
#  pMatches.wait()
  
#if 1:  
#  print ("3. Do Incremental\\Sequential reconstruction") #set manually the initial pair to avoid the prompt question
#  pRecons = subprocess.Popen( [str(OPENMVG_SFM_BIN.joinpath("openMVG_main_IncrementalSfM")),  "-i", str(matches_dir.joinpath("sfm_data.json")), "-m", matches_dir, "-o",reconstruction_dir,"-P"])#,"-f","ADJUST_ALL","-c","3"] )
#  pRecons.wait()

  
#  print ("3. Do Incremental\\Sequential reconstruction") #set manually the initial pair to avoid the prompt question
#  pRecons = subprocess.Popen( [ str(OPENMVG_SFM_BIN.joinpath("openMVG_main_IncrementalSfM2")),  "-i", str(matches_dir.joinpath("sfm_data.json")), "-m", matches_dir, "-o", reconstruction_dir2,"-P"])#,"-f","ADJUST_ALL","-c","3"] )
#  pRecons.wait()
#if 1:   
#  print ("5. Exports")
#  pRecons = subprocess.Popen( [ str(OPENMVG_SFM_BIN.joinpath("openMVG_main_openMVG2openMVS")),  "-i", str(reconstruction_dir.joinpath("sfm_data.bin")), "-o", str(reconstruction_dir_scene.joinpath("scene.mvs")), "-d", str(reconstruction_dir_scene.joinpath("undistorted"))] )
#  pRecons.wait()
    
  
#  print ("5. Exports")
#  pRecons = subprocess.Popen( [ str(OPENMVG_SFM_BIN.joinpath("openMVG_main_openMVG2openMVS")),  "-i", str(reconstruction_dir2.joinpath("sfm_data.bin")), "-o", str(reconstruction_dir_scene2.joinpath("scene.mvs")), "-d", str(reconstruction_dir_scene2.joinpath("undistorted"))] )
#  pRecons.wait()
  

if 1:   
  print ("6. Dense")
  pRecons = subprocess.Popen( [ str(OPENMVS_BIN.joinpath("DensifyPointCloud")), str(reconstruction_dir_scene.joinpath("scene.mvs")), "--estimate-normals","1","--number-views","0","-v","3"])#,"--number-views-fuse","5"] )
  pRecons.wait()

  
  pRecons = subprocess.Popen( [ str(OPENMVS_BIN.joinpath("DensifyPointCloud")), str(reconstruction_dir_scene2.joinpath("scene.mvs")), "--estimate-normals","1","--number-views","0","-v","3"])#,"--number-views-fuse","5"] )
  pRecons.wait()
  
##if 0: 
  print ("7. Mesh")
  pRecons = subprocess.Popen( [str(OPENMVS_BIN.joinpath("ReconstructMesh")), str(reconstruction_dir_scene.joinpath("scene_dense.mvs"))] )
  pRecons.wait()
 
  pRecons = subprocess.Popen( [str(OPENMVS_BIN.joinpath("ReconstructMesh")), str(reconstruction_dir2_scene.joinpath("scene_dense.mvs"))] )
  pRecons.wait()
  

  
  print ("8. Refine Mesh")
  
  import shutil
  try:
    shutil.copyfile(str(reconstruction_dir_scene.joinpath("scene_dense_mesh.mvs")), str(reconstruction_dir_scene.joinpath("scene_dense_mesh_refined.mvs")))
  except:
    pass
    
  try:
    shutil.copyfile(str(reconstruction_dir_scene2.joinpath("scene_dense_mesh.mvs")), str(reconstruction_dir_scene2.joinpath("scene_dense_mesh_refined.mvs")))
  except:
    pass

  try:
    shutil.copyfile(str(reconstruction_dir_scene.joinpath("scene_robust_dense_mesh.mvs")), str(reconstruction_dir_scene.joinpath("scene_robust_dense_mesh_refined.mvs")))
  except:
    pass

if 0:
  pRecons = subprocess.Popen( [str(OPENMVS_BIN.joinpath("RefineMesh")), str(reconstruction_dir_scene.joinpath("scene_dense_mesh_refined.mvs")),"--use-cuda", "0"] )
  #print(str(pRecons))
  #print("the commandline is {}".format(pRecons.args))
  pRecons.wait()
  pRecons = subprocess.Popen( [str(OPENMVS_BIN.joinpath("RefineMesh")), str(reconstruction_dir_scene2.joinpath("scene_dense_mesh_refined.mvs")),"--use-cuda", "0"] )
  #print(str(pRecons))
  #print("the commandline is {}".format(pRecons.args))
  pRecons.wait()
if 1:  
  print ("9. Texture")
  pRecons = subprocess.Popen( [str(OPENMVS_BIN.joinpath("TextureMesh")), str(reconstruction_dir_scene.joinpath("scene_dense_mesh.mvs")),"--export-type", "obj"] )
  pRecons.wait()
  pRecons = subprocess.Popen( [str(OPENMVS_BIN.joinpath("TextureMesh")), str(reconstruction_dir_scene.joinpath("scene_dense_mesh_refined.mvs")),"--export-type", "obj"] )
  pRecons.wait()
  pRecons = subprocess.Popen( [str(OPENMVS_BIN.joinpath("TextureMesh")), str(reconstruction_dir_scene2.joinpath("scene_dense_mesh.mvs")),"--export-type", "obj"] )
  pRecons.wait()
  pRecons = subprocess.Popen( [str(OPENMVS_BIN.joinpath("TextureMesh")), str(reconstruction_dir_scene2.joinpath("scene_dense_mesh_refined.mvs")),"--export-type", "obj"] )
  pRecons.wait()

