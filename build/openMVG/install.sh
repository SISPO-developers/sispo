#!/bin/bash

echo "Installing openMVG start"

# create dir
cd ../..
[[ -d software ]] || mkdir software
cd software

[[ -d openMVG ]] || mkdir openMVG
cd openMVG

# Clone git repo
git clone --recursive https://github.com/openMVG/openMVG.git

# Building
[[ -d build_openMVG ]] || mkdir build_openMVG
cd build_openMVG

cmake \
	-S ../openMVG/src \
        -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DCMAKE_INSTALL_PREFIX=install/ \
        -DINCLUDE_INSTALL_DIR=install/include \
        -DPYTHON_EXECUTABLE=../../conda/envs/py37/bin/python

cmake --build . --target install

echo "Installing openMVG done"
