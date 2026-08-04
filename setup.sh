#!/bin/sh
# Launches the setup program, which writes ~/.buzz/config.toml.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$DIR/lib" python3 -m buzz.setup
