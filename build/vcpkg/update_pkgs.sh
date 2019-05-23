#!/bin/bash

echo "Updating all vcpkg packages start"

# Change to vcpkg dir
cd ../../software/vcpkg

# Update and upgrade packages
./vcpkg update
./vcpkg upgrade
./vcpkg upgrade --no-dry-run

echo "Updating all vcpkg packages done"
