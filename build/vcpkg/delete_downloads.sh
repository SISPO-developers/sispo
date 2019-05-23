#!/bin/bash

echo "Deleting downloaded and cached binaries start"

cd ../../software/vcpkg/downloads

rm *.tar*
rm *.zip*

echo "Deleting downloaded and cached binaries done"
