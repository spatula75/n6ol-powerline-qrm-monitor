#!/bin/sh
# Runs the monitor with no special arguments, using the .venv setup.sh created.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
CONFIG_DIR="$HOME/.buzz"
CONFIG_FILE="$CONFIG_DIR/config.toml"

# See setup.sh's identical helper: a venv's interpreter sits at bin/python on a
# real POSIX layout, but at Scripts/python.exe when a native Windows Python
# created it - which happens even under a POSIX shell like this one in Git Bash.
venv_python() {
    if [ -x "$1/bin/python" ]; then
        echo "$1/bin/python"
    elif [ -x "$1/Scripts/python.exe" ]; then
        echo "$1/Scripts/python.exe"
    fi
}

PYTHON="$(venv_python "$VENV")"
if [ -z "$PYTHON" ]; then
    echo "No virtual environment found at $VENV. Run ./setup.sh first." >&2
    exit 1
fi

# The file, not just the directory: FinishScreen.save() creates ~/.buzz right
# before writing config.toml into it, so a setup run that crashed or was killed
# between those two steps leaves the directory behind with no config inside it.
# Starting on defaults nobody chose in that state is exactly what this check
# exists to refuse.
if [ ! -f "$CONFIG_FILE" ]; then
    echo "No configuration found at $CONFIG_FILE. Run ./setup.sh first." >&2
    exit 1
fi

PYTHONPATH="$DIR/lib" "$PYTHON" -m buzz.main
