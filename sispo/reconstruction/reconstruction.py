#!/usr/bin/python
#!  -*- encoding: utf-8 -*-

# This file is part of OpenMVG (Open Multiple View Geometry) C++ library.

# Python script to launch OpenMVG SfM tools on an image dataset
#
# usage : python tutorial_demo.py
#

import reconstruction.openmvg as openmvg

#from os import system
import os
#import sys
from pathlib import Path
from subprocess import Popen

#SOFTWARE_DIR = Path.cwd().joinpath("software")


#system("title " + '"%s"' % (os.path.abspath(__file__)))
#OPENMVG_SFM_BIN =
#SOFTWARE_DIR.joinpath("openMVG").joinpath("build").joinpath("Windows-AMD64-").joinpath("Release")
#OPENMVS_BIN =
#SOFTWARE_DIR.joinpath("openMVS").joinpath("build").joinpath("bin").joinpath("x64").joinpath("Debug")
## Indicate the openMVG camera sensor width directory
#CAMERA_SENSOR_WIDTH_DIRECTORY =
#SOFTWARE_DIR.joinpath("openMVG").joinpath("openMVG").joinpath("src").joinpath("openMVG").joinpath("exif").joinpath("sensor_width_database")

#root = Path.cwd()
#workdir = root.joinpath("data").joinpath("results")
#if not workdir.exists():
#  Path.mkdir(workdir)
  
##os.chdir(workdir)
#input_eval_dir =
#Path.cwd().joinpath("data").joinpath("ImageDataset_SceauxCastle-master")
## Checkout an OpenMVG image dataset with Git
#output_eval_dir = input_eval_dir.joinpath("reconstruction")
##output_eval_dir = os.path.join(get_parent_dir(input_eval_dir),
##"asteroid_out")
#input_eval_dir = input_eval_dir.joinpath("images")
#if not input_eval_dir.exists():
#  Path.mkdir(input_eval_dir)

#if not output_eval_dir.exists():
#  Path.mkdir(output_eval_dir)

#input_dir = input_eval_dir
#output_dir = output_eval_dir
#print("Using input dir : ", input_dir)
#print(" output_dir : ", output_dir)

#matches_dir = output_dir.joinpath("matches")
#camera_file_params =
#CAMERA_SENSOR_WIDTH_DIRECTORY.joinpath("sensor_width_camera_database.txt")

## Create the ouput\\matches folder if not present
#if not matches_dir.exists():
#  Path.mkdir(matches_dir)

#fl = 65437
#reconstruction_dir = output_dir.joinpath("reconstruction_sequential")
#if not reconstruction_dir.exists():
#  Path.mkdir(reconstruction_dir)

#reconstruction_dir_scene = reconstruction_dir.joinpath("scene")
#if not reconstruction_dir_scene.exists():
#  Path.mkdir(reconstruction_dir_scene)

#reconstruction_dir2 = output_dir.joinpath("reconstruction_sequential2")
#if not reconstruction_dir2.exists():
#  Path.mkdir(reconstruction_dir2)

#reconstruction_dir_scene2 = reconstruction_dir2.joinpath("scene")
#if not reconstruction_dir_scene2.exists():
#  Path.mkdir(reconstruction_dir_scene2)

#if 1:
#  print ("1.  Intrinsics analysis")
#  pIntrisics = Popen(
#  [str(OPENMVG_SGM_BIN.joinpath("openMVG_main_SfMInit_ImageListing")), "-i",
#  input_dir, "-o", str(matches_dir), "-d", str(camera_file_params), "-c",
#  "1","-f",str(fl),"-P","-W","1.0;1.0;1.0;"] )
#  pIntrisics.wait()
  
#  print ("2.  Compute features")
#  pFeatures = Popen( [
#  str(OPENMVG_SFM_BIN.joinpath("openMVG_main_ComputeFeatures")), "-i",
#  str(matches_dir.joinpath("sfm_data.json")), "-o", str(matches_dir), "-m",
#  "SIFT", "-f" , "0","-p","ULTRA"] )
#  pFeatures.wait()
#if 1:
#  print ("2.  Compute matches")
#  pMatches = Popen( [
#  str(OPENMVG_SFM_BIN.joinpath("openMVG_main_ComputeMatches")), "-i",
#  str(matches_dir.joinpath("sfm_data.json")), "-o", matches_dir, "-f", "1",
#  "-n", "FASTCASCADEHASHINGL2","-v","12"] )
#  pMatches.wait()
  
#if 1:
#  print ("3.  Do Incremental\\Sequential reconstruction") #set manually the
#  initial pair to avoid the prompt question
#  pRecons = Popen(
#  [str(OPENMVG_SFM_BIN.joinpath("openMVG_main_IncrementalSfM")), "-i",
#  str(matches_dir.joinpath("sfm_data.json")), "-m", matches_dir,
#  "-o",reconstruction_dir,"-P"])#,"-f","ADJUST_ALL","-c","3"] )
#  pRecons.wait()

  
#  print ("3.  Do Incremental\\Sequential reconstruction") #set manually the
#  initial pair to avoid the prompt question
#  pRecons = Popen( [
#  str(OPENMVG_SFM_BIN.joinpath("openMVG_main_IncrementalSfM2")), "-i",
#  str(matches_dir.joinpath("sfm_data.json")), "-m", matches_dir, "-o",
#  reconstruction_dir2,"-P"])#,"-f","ADJUST_ALL","-c","3"] )
#  pRecons.wait()
#if 1:
#  print ("5.  Exports")
#  pRecons = Popen( [
#  str(OPENMVG_SFM_BIN.joinpath("openMVG_main_openMVG2openMVS")), "-i",
#  str(reconstruction_dir.joinpath("sfm_data.bin")), "-o",
#  str(reconstruction_dir_scene.joinpath("scene.mvs")), "-d",
#  str(reconstruction_dir_scene.joinpath("undistorted"))] )
#  pRecons.wait()
    
  
#  print ("5.  Exports")
#  pRecons = Popen( [
#  str(OPENMVG_SFM_BIN.joinpath("openMVG_main_openMVG2openMVS")), "-i",
#  str(reconstruction_dir2.joinpath("sfm_data.bin")), "-o",
#  str(reconstruction_dir_scene2.joinpath("scene.mvs")), "-d",
#  str(reconstruction_dir_scene2.joinpath("undistorted"))] )
#  pRecons.wait()
  
#if 1:
#  print("6.  Dense")
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("DensifyPointCloud")),
#  str(reconstruction_dir_scene.joinpath("scene.mvs")),
#  "--estimate-normals","1","--number-views","0","-v","3"])#,"--number-views-fuse","5"]
#  )
#  pRecons.wait()

  
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("DensifyPointCloud")),
#  str(reconstruction_dir_scene2.joinpath("scene.mvs")),
#  "--estimate-normals","1","--number-views","0","-v","3"])#,"--number-views-fuse","5"]
#  )
#  pRecons.wait()
  
###if 0:
#  print("7.  Mesh")
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("ReconstructMesh")),
#  str(reconstruction_dir_scene.joinpath("scene_dense.mvs"))])
#  pRecons.wait()
 
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("ReconstructMesh")),
#  str(reconstruction_dir2_scene.joinpath("scene_dense.mvs"))])
#  pRecons.wait()
  

  
#  print("8.  Refine Mesh")
  
#  import shutil
#  try:
#    shutil.copyfile(str(reconstruction_dir_scene.joinpath("scene_dense_mesh.mvs")),
#    str(reconstruction_dir_scene.joinpath("scene_dense_mesh_refined.mvs")))
#  except:
#    pass
    
#  try:
#    shutil.copyfile(str(reconstruction_dir_scene2.joinpath("scene_dense_mesh.mvs")),
#    str(reconstruction_dir_scene2.joinpath("scene_dense_mesh_refined.mvs")))
#  except:
#    pass

#  try:
#    shutil.copyfile(str(reconstruction_dir_scene.joinpath("scene_robust_dense_mesh.mvs")),
#    str(reconstruction_dir_scene.joinpath("scene_robust_dense_mesh_refined.mvs")))
#  except:
#    pass

#if 0:
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("RefineMesh")),
#  str(reconstruction_dir_scene.joinpath("scene_dense_mesh_refined.mvs")),"--use-cuda",
#  "0"])
#  #print(str(pRecons))
#  #print("the commandline is {}".format(pRecons.args))
#  pRecons.wait()
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("RefineMesh")),
#  str(reconstruction_dir_scene2.joinpath("scene_dense_mesh_refined.mvs")),"--use-cuda",
#  "0"])
#  #print(str(pRecons))
#  #print("the commandline is {}".format(pRecons.args))
#  pRecons.wait()
#if 1:
#  print("9.  Texture")
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("TextureMesh")),
#  str(reconstruction_dir_scene.joinpath("scene_dense_mesh.mvs")),"--export-type",
#  "obj"])
#  pRecons.wait()
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("TextureMesh")),
#  str(reconstruction_dir_scene.joinpath("scene_dense_mesh_refined.mvs")),"--export-type",
#  "obj"])
#  pRecons.wait()
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("TextureMesh")),
#  str(reconstruction_dir_scene2.joinpath("scene_dense_mesh.mvs")),"--export-type",
#  "obj"])
#  pRecons.wait()
#  pRecons = Popen([str(OPENMVS_BIN.joinpath("TextureMesh")),
#  str(reconstruction_dir_scene2.joinpath("scene_dense_mesh_refined.mvs")),"--export-type",
#  "obj"])
#  pRecons.wait()
class Reconstructor():
    """Reconstruction of a 3D object from images."""

    def __init__(self):
        """Initialises main directory and file structure."""
        root_dir = Path(__file__).parent.parent.parent
        res_dir = root_dir / "data" / "results" / "Didymos"
        oMVG = openmvg.OpenMVGController(res_dir)
        oMVG.analyse_images()
        oMVG.compute_features()
        oMVG.match_features()
        oMVG.reconstruct_seq()
        oMVG.export_MVS()

        #file_dir = Path(__file__)
        #root_dir = Path(file_dir / ".." / ".." / "..").resolve()
        #software_dir = root_dir / "software"
        #data_dir = root_dir / "data"
#
        #self.openMVG_dir = software_dir / "openMVG" / "build_openMVG" / "Windows-AMD64-Release" / "Release"
        #self.openMVS_dir = software_dir / "openMVS" / "build" / "bin" / "x64" / "Debug"
        #self.sensor_database_dir = data_dir / "sensor_database"
#
        #self.input_dir = data_dir / "ImageDataset_SceauxCastle-master" / "images"
        #self.output_dir = self._resolve_create_dir(data_dir / "results" / "reconstruction")
#
        #self.sensor_database = self.sensor_database_dir / "sensor_width_camera_database.txt"
#
        #self.fl = 65437

    @staticmethod
    def _resolve_create_dir(directory):
        """Resolves directory and creates it, if it doesn't existing."""
        dir_resolved = directory.resolve()

        if not dir_resolved.exists():
            Path.mkdir(dir_resolved)

        return dir_resolved
        
    def analyse_intrinsically(self):
        """ImageListing step of reconstruction."""
        print("1. Intrinsics analysis")

        #self.features_dir = self._resolve_create_dir(self.output_dir /
        #"features")
        self.matches_dir = self._resolve_create_dir(self.output_dir / "matches")

        pIntrisics = Popen([str(self.openMVG_dir / "openMVG_main_SfMInit_ImageListing"),
                            "-i", str(self.input_dir), 
                            "-o", str(self.matches_dir),
                            "-d", str(self.sensor_database),
                            "-c", "1",
                            "-f", str(self.fl), 
                            "-P", 
                            "-W", "1.0;1.0;1.0;"])
        pIntrisics.wait()

    def compute_features(self):
        """Compute features in the pictures."""
        print("2. Compute features")

        
        self.sfm_data = self.matches_dir / "sfm_data.json"

        pFeatures = Popen([str(self.openMVG_dir / "openMVG_main_ComputeFeatures"),
                           "-i", str(self.sfm_data),
                           "-o", str(self.matches_dir), 
                           "-m", "SIFT",
                           "-f", "0", 
                           "-p", "ULTRA"])
        pFeatures.wait()

    def compute_matches(self):
        """Compute feature matches."""
        print("3. Compute matches")

        print(str(self.sfm_data))
        print(str(self.matches_dir))

        pMatches = Popen([str(self.openMVG_dir / "openMVG_main_ComputeMatches"),
                          "-i", str(self.sfm_data),
                          "-o", str(self.matches_dir), 
                          "-f", "1", 
                          "-n", "FASTCASCADEHASHINGL2",
                          "-v", "12"])
        pMatches.wait()

    def reconstruct_sequentially(self):
        """Reconstruct 3D models sequentially."""
        #set manually the initial pair to avoid the prompt question
        print("4. Do Incremental\\Sequential reconstruction")

        self.reconstruction_dir = self._resolve_create_dir(self.output_dir / "sequential")

        pRecons = Popen([str(self.openMVG_dir / "openMVG_main_IncrementalSfM"),
                         "-i", str(self.sfm_data),
                         "-m", str(self.matches_dir), 
                         "-o", str(self.reconstruction_dir),
                         "-P"])#,"-f","ADJUST_ALL","-c","3"] )
        pRecons.wait()

    def export_MVG2MVS(self):
        """Export 3D model to MVS format."""
        print("5. Exports")

        self.export_dir = self._resolve_create_dir(self.output_dir / "export")
        self.export_scene = self.export_dir / "scene.mvs"
        self.export_undistorted_dir = self._resolve_create_dir(self.export_dir / "undistorted")

        pExport = Popen([str(self.openMVG_dir / "openMVG_main_openMVG2openMVS"),
                         "-i", str(self.reconstruction_dir / "sfm_data.bin"),
                         "-o", str(self.export_scene),
                         "-d", str(self.export_undistorted_dir)])
        pExport.wait()

    def densify_pointcloud(self):
        """Increases number of points to make 3D model smoother."""
        print("6. Dense")
        pDensify = Popen([str(self.openMVS_dir / "DensifyPointCloud"),
                          str(self.export_scene),
                          "--max-threads", "0",
                          "--estimate-normals", "1",
                          "--number-views", "0",
                          "-v", "3"])#,"--number-views-fuse","5"] )
        pDensify.wait()

    def create_mesh(self):
        """Create a mesh from a 3D point cloud."""
        print("7. Mesh")

        self.export_scene_dense = (self.output_dir / "export" / "scene_dense.mvs").resolve()

        pMesh = Popen([str(self.openMVS_dir / "ReconstructMesh"),
                       str(self.export_scene_dense)])
        pMesh.wait()

    def refine_mesh(self):
        """Refine 3D mesh."""
        print("8. Refine Mesh")

        self.export_scene_dense_mesh = (self.output_dir / "export" / "scene_dense_mesh.mvs").resolve()

        pRefineMesh = Popen([str(self.openMVS_dir / "RefineMesh"),
                             str(self.export_scene_dense_mesh),
                             "--use-cuda", "0"])
        pRefineMesh.wait()

    def texture_mesh(self):
        """Put texture on mesh model using pictures."""
        print("9. Texture")

        self.export_scene_dense_mesh_refined = (self.output_dir / "export" / "scene_dense_mesh_refined.mvs").resolve()

        pTexture1 = Popen([str(self.openMVS_dir / "TextureMesh"),
                           str(self.export_scene_dense_mesh),
                           "--export-type", "obj"])

        pTexture2 = Popen([str(self.openMVS_dir / "TextureMesh"),
                           str(self.export_scene_dense_mesh_refined),
                           "--export-type", "obj"])
        pTexture1.wait()
        pTexture2.wait()

if __name__ == "__main__":
    recon = Reconstructor()
    #recon.analyse_intrinsically()
    #recon.compute_features()
    #recon.compute_matches()
    #recon.reconstruct_sequentially()
    #recon.export_MVG2MVS()
    #recon.densify_pointcloud()
    recon.create_mesh()
    recon.refine_mesh()
    recon.texture_mesh()
