#!/bin/bash

echo "Downloading orekit-data.zip start"

# Create folder
cd ../..
[[ -d data ]] || mkdir data
cd data

# Download data
wget https://gitlab.orekit.org/orekit/orekit-data/-/archive/master/orekit-data-master.zip
mv orekit-data-master.zip orekit-data.zip

echo "Downloading orekit-data.zip done"
