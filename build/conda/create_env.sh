#!/bin/bash

echo "Creating conda environment sispo start"

# Create
echo y | conda create env -f ../../environment.yml

echo "Creating conda environment sispo done"
