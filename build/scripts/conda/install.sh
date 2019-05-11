#!/bin/bash

echo "Installing miniconda3 start"

# Creating miniconda dir
cd ../..
mkdir software
cd software

# Getting latest miniconda3 binaries
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Installing
./Miniconda3-latest-Linux-x86_64.sh

# Deleting binaries
rm Miniconda3-latest-Linux-x86_64.sh

# Appending conda-forge channel
conda config --append channels conda-forge

echo "Installing miniconda3 done"
