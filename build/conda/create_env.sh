#!/bin/bash

# SPDX-FileCopyrightText: 2021 Gabriel J. Schwarzkopf <sispo-devs@outlook.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

echo "Creating conda environment sispo start"

# Create
echo y | conda create env -f ../../environment.yml

echo "Creating conda environment sispo done"
