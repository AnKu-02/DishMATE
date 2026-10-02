#!/usr/bin/env bash
# Existing local SDK network settings; override these for another installation.
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_wsl.sh"
timeout 25s python "$SCRIPT_DIR/main_real.py" \
    --camera-only --ros --timeout 10 "$@"
