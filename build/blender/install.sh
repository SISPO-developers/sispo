#!/bin/bash

echo "Installing blender as a python module start"

# Creating blender directory
cd ../.. || exit
[[ -d software ]] || mkdir software
cd software || exit

[[ -d blender ]] || mkdir blender
cd blender || exit

# Get blender repo
git clone https://git.blender.org/blender.git
cd blender || exit
git submodule update --init --recursive
git submodule foreach git checkout master
git submodule foreach pull --rebase origin master

# Update files
make update

###########################
# Get stable bpy version
#cd ../../../build/blender
#./../../../build/blender/checkout_stable_bpy.sh
#cd ../../software/blender/blender
###########################

# Install blender dependenciesn
#./build_files/build_environment/install_deps.sh

# Configure and install blender bpy
cd .. || exit
[[ -d build_blender_bpy ]] || mkdir build_blender_bpy
cd build_blender_bpy || exit
cmake \
	-C ../blender/build_files/cmake/config/bpy_module.cmake \
	-S ../blender \
	-DWITH_CYCLES_CUDA_BINARIES=ON \
	-DWITH_IMAGE_OPENEXR=ON \
	-DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake \
	-DPYTHON_SITE_PACKAGES=../../conda/envs/sispo/lib/python3.7/site-packages
cmake --build . 
make install

echo "Installing blender as a python module done"
