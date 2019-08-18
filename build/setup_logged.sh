#!/bin/bash
set -euo pipefail
exec 2>&1

echo "Setup with logging what's shappening start"

./setup.sh | tee setup.log

echo "Setup with logging what's happening done"
