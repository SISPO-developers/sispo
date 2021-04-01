#!/bin/bash

# SPDX-FileCopyrightText: 2021 Gabriel J. Schwarzkopf <sispo-devs@outlook.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

echo "Updating all vcpkg packages start"

# Change to vcpkg dir
cd ../../software/vcpkg || exit

# Update and upgrade packages
./vcpkg update
./vcpkg upgrade
./vcpkg upgrade --no-dry-run

echo "Updating all vcpkg packages done"
