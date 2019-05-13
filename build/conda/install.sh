#!/bin/bash

echo "Installing miniconda3 start"

# Creating miniconda dir
cd ../..
[[ -d software ]] || mkdir software
cd software

# Getting latest miniconda3 binaries
[[ -d Miniconda3-latest-Linux-x86_64.sh ]] || wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Installing
./Miniconda3-latest-Linux-x86_64.sh -b -p conda -f
cd conda/bin
./conda init bash
./conda config --append channels conda-forge

# Deleting binaries
cd ../..
rm Miniconda3-latest-Linux-x86_64.sh

echo "Installing miniconda3 done"
