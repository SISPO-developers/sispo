#!/bin/bash

echo "Setup system for MasterThesis project"

( (
echo "########## conda start ##########"
cd conda || exit
./install.sh
./update.sh

#./create_env.sh
#./activate_env.sh

./install_pkgs.sh
./update_pkgs.sh

cd .. || exit
echo "########## conda done ##########"
) &

(
echo "########## vcpkg start ##########"
cd vcpkg || exit
./install.sh
./update.sh

./install_pkgs.sh
./update_pkgs.sh

./delete_downloads.sh
cd .. || exit
echo "########## vcpkg done ##########"
) &

(
echo "########## star_cats start ##########"
cd star_cats || exit
./install.sh

cd .. || exit
echo "########## star_cats done ##########"
) &

#########################################
wait
#########################################

(
echo "########## bpy start ##########"
cd blender || exit
./install.sh

cd .. || exit
echo "########## bpy done ##########"
) &

(
echo "########## openMVG start ##########"
cd openMVG || exit
./install.sh

cd .. || exit
echo "########## openMVG done ##########"
) &

(
echo "########## openMVS start ##########"
cd openMVS || exit
./install.sh

cd .. || exit
echo "########## openMVS done ##########"
) & ) &

(
echo "########## download data start ##########"
cd data || exit
#./download_orekit.sh
#./download_ucac4.sh

cd .. || exit
echo "########## download data done ##########"
) &

echo "Setup complete"
