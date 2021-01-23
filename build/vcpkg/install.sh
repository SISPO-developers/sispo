#!/bin/bash

echo "vcpkg install start"

# Go to automation root
cd ../../ || exit

# Create software dir
mkdir software || exit
cd software || exit

# Get vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg || exit

# Install and integrate
./bootstrap-vcpkg.sh
./vcpkg integrate install

echo "vcpkg install done"
