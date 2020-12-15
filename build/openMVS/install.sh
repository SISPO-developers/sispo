#!/bin/bash

echo "Installing openMVS start"

# Create dir
cd ../..
[[ -d software ]] || mkdir software
cd software

[[ -d openMVS ]] || mkdir openMVS
cd openMVS

# Clone git repo
[[ -d VCG ]] || git clone https://github.com/cdcseacave/VCG.git
[[ -d openMVS ]] || git clone https://github.com/cdcseacave/openMVS.git openMVS

# MBuild openMVS
[[ -d build_openMVS ]] || mkdir build_openMVS
cd build_openMVS
cmake \
	-S ../openMVS \
	-DCMAKE_BUILD_TYPE=Release \
	-DVCG_DIR=../VCG \
	-DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_INSTALL_PREFIX=install
cmake --build . --target install

echo "Installing openMVS done"

