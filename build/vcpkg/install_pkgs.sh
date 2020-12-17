#!/bin/bash

echo "Installing required packages start"

# Go into vcpkg dir
cd ../../software/vcpkg || exit

# Install packages
# General packages
./vcpkg install \
	--recurse \
	--keep-going \
       eigen3 \
       glog \
       gflags \
       zlib \
       bzip2 \
       vcglib \
       tbb \
       boost \
       openblas \
       clapack \
       ceres[lapack,tools] \
       glew \
       glfw3 \
       mpir \
       mpfr \
       pthreads \
       openexr \
       cereal \
       cgal \
       tiff \
       libjpeg-turbo \
       libpng \
       sdl2 \
       ogre \
       opencv[contrib,dnn,eigen,ffmpeg,flann,ipp,jasper,jpeg,openexr,opengl,png,sfm,tiff]
       #qt5 \
       #intel-mkl \
       #breakpad \


echo "Installing required packages done"
