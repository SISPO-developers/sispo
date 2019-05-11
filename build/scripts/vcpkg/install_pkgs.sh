#!/bin/bash

echo "Installing required packages start"

# Go into vcpkg dir
cd ../../software/vcpkg

# Install packages
# General packages
./vcpkg install \
       eigen3 \
       glog \
       gflags \
       zlib \
       vcglib \
       boost \
       atlmfc \
       bzip2 \
       ceres \
       glew \
       glfw3 \
       ogre \
       openexr \
       cereal \
       tiff \
       libjpeg-turbo \
       libpng \
       cgal \
       qt5 \
       opencv[contrib,dnn,eigen,ffmpeg,flann,ipp,jasper,jpeg,openexr,opengl,ovis,png,qt,sfm,tbb,tiff]

echo "Installing required packages done"
