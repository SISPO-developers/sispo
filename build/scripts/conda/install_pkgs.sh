#!/bin/bash

echo "Installing conda packages start"

conda install \
	pip \
	setuptools \
	wheel \
	numpy \
	matplotlib \
	flake8 \
	pylint \
	autopep8 \
	sphinx \
	zlib \
	scikit-image \
	scipy \
	simplejson\
	orekit

echo "Installing conda packages done"
