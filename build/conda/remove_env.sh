#!/bin/bash

# SPDX-FileCopyrightText: 2021 Gabriel J. Schwarzkopf <sispo-devs@outlook.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

echo "Removing conda environment py37 start"

# Remove
conda env remove --name sispo

echo "Removing conda environment py37 done"
