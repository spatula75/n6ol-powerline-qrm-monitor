#!/bin/sh
# Bootstraps the environment, then launches the setup program, which writes
# ~/.buzz/config.toml.
#
# Reuses .venv if it already exists and is Python 3.12 or later; otherwise finds a
# system Python that is, creates .venv with it, and installs requirements.txt.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"

is_312_or_later() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1
}

# A venv's interpreter sits at bin/python on a real POSIX layout, but at
# Scripts/python.exe when a native Windows Python created it - which happens even
# under a POSIX shell like this one running in Git Bash.  Check both rather than
# assume the layout matches the shell.
venv_python() {
    if [ -x "$1/bin/python" ]; then
        echo "$1/bin/python"
    elif [ -x "$1/Scripts/python.exe" ]; then
        echo "$1/Scripts/python.exe"
    fi
}

find_system_python() {
    for candidate in python3.14 python3.13 python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && is_312_or_later "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON="$(venv_python "$VENV")"
if [ -z "$PYTHON" ] || ! is_312_or_later "$PYTHON"; then
    if [ -n "$PYTHON" ]; then
        echo "Existing .venv is older than Python 3.12; recreating it."
    fi
    SYSTEM_PYTHON="$(find_system_python)" || {
        echo "No Python 3.12 or later found on PATH. Install Python 3.12+ and try again." >&2
        exit 1
    }
    echo "Creating virtual environment at $VENV using $SYSTEM_PYTHON..."
    "$SYSTEM_PYTHON" -m venv "$VENV"
    PYTHON="$(venv_python "$VENV")"
fi

echo "Installing requirements..."
"$PYTHON" -m pip install -r "$DIR/requirements.txt"

PYTHONPATH="$DIR/lib" "$PYTHON" -m buzz.setup
