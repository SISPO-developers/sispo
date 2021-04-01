#!/bin/bash

# SPDX-FileCopyrightText: 2021 Gabriel J. Schwarzkopf <sispo-devs@outlook.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

echo "Deleting downloaded and cached binaries start"

cd ../../software/vcpkg/downloads || exit

rm -- *.tar*
rm -- *.zip*

echo "Deleting downloaded and cached binaries done"
