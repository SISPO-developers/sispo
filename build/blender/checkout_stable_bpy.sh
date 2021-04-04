#!/bin/bash

# SPDX-FileCopyrightText: 2021 Gabriel J. Schwarzkopf <sispo-devs@outlook.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

echo "Checking out stable blender version with functioning bpy start"

# Commit e2102e9 supposed to be stable, see blender T64427 task
git checkout e2102e9

echo "Checking out stable blender version with functioning bpy done"
