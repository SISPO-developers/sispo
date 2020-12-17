#!/bin/bash

echo "Installing star catalogue library start"

# Create dir
cd ../.. || exit
[[ -d software ]] || mkdir software
cd software || exit

[[ -d star_cats ]] || mkdir star_cats
cd star_cats || exit

# Cloning git repo
git clone https://github.com/Bill-Gray/star_cats.git

# Install
[[ -d build_star_cats ]] || mkdir build_star_cats
cd build_star_cats || exit

cp ../star_cats/* .
make
rm *.* LICENSE makefile

echo "Installing star catalogue library done"
