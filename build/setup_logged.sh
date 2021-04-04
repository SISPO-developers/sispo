#!/bin/bash

# SPDX-FileCopyrightText: 2021 Gabriel J. Schwarzkopf <sispo-devs@outlook.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail
exec 2>&1

echo "Setup with logging what's shappening start"

./setup.sh | tee setup.log

echo "Setup with logging what's happening done"
