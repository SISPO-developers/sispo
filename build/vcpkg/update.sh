#!/bin/bash

# SPDX-FileCopyrightText: 2021 Gabriel J. Schwarzkopf <sispo-devs@outlook.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

echo "Updating vcpkg start"

# Get latest git
cd ../../software/vcpkg || exit
git pull

# Install and integrate
./bootstrap-vcpkg.sh
./vcpkg integrate install

echo "Updating vcpkg done"
