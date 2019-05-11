#!/bin/bash

echo "Installing openMVS start"

# Create dir
cd ../..
[[ -d software ]] || mkdir software
cd software

[[ -d openMVS ]] || mkdir openMVS
cd openMVS

# Clone git repo
git clone https://github.com/cdcseacave/VCG.git
git clone https://github.com/cdcseacave/openMVS.git openMVS

# MBuild openMVS
[[ -d build_openMVS ]] || mkdir build_openMVS
cd build_openMVS
cmake \
	-S ../openMVS \
	-B . \
	-DCMAKE_BUILD_TYPE=Release \
	-DVCG_ROOT=../VCG \
	-DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -j 8
make install --target install

echo "Installing openMVS done"

