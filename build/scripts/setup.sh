#!/bin/bash

echo "Setup system for MasterThesis project"

echo "########## conda start ##########"
cd conda
./install.sh
./update.sh

./create_env.sh
./activate_env.sh

./install_pkgs.sh
./update_pkgs.sh

cd ..
echo "########## conda done ##########"

echo "########## vcpkg start ##########"
cd vcpkg
./install.sh
./update.sh

./install_pkgs.sh
./update_pkgs.sh

cd ..
echo "########## vcpkg done ##########"

echo "########## bpy start ##########"
cd blender
./install.sh

cd ..
echo "########## bpy done ##########"

echo "########## openMVG start ##########"
cd openMVG
./install.sh

cd ..
echo "########## openMVG done ##########"

echo "########## openMVS start ##########"
cd openMVS
./install.sh

cd ..
echo "########## openMVS done ##########"

echo "Setup complete"
