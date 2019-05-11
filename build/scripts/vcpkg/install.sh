#!/bin/bash

echo "vcpkg install start"

# Go to automation root
cd ../../

# Create software dir
mkdir software
cd software

# Get vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Install and integrate
./bootstrap-vcpkg.sh
./vcpkg integrate install

echo "vcpkg install done"
