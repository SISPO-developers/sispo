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
       bzip2 \
       vcglib \
       tbb \
       #intel-mkl \
       #boost[mpi] \
       #breakpad \
       #ceres[lapack,tools] \
       #glew \
       #glfw3 \
       #ogre \
       #openexr \
       #cereal \
       #cgal \
       #tiff \
       #libjpeg-turbo \
       #libpng \
       #opencv[contrib,dnn,eigen,ffmpeg,flann,ipp,jasper,jpeg,openexr,opengl,ovis,png,qt,sfm,tbb,tiff]
       #atlmfc \
       #qt5 \


echo "Installing required packages done"
