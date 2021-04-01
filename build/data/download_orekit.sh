#!/bin/bash

# SPDX-FileCopyrightText: 2021 Gabriel J. Schwarzkopf <sispo-devs@outlook.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

echo "Downloading orekit-data.zip start"

# Create folder
cd ../.. || exit
[[ -d data ]] || mkdir data
cd data || exit

# Download data
wget https://gitlab.orekit.org/orekit/orekit-data/-/archive/master/orekit-data-master.zip
mv orekit-data-master.zip orekit-data.zip

echo "Downloading orekit-data.zip done"
