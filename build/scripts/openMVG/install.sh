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
[[ -d openMVG_build ]] || mkdir openMVG_build
cd openMVG_build

cmake \
	../openMVG/src \
       -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build . --target install

echo "Installing openMVG done"
