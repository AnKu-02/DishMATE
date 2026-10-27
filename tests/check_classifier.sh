#!/usr/bin/env bash
# Uses Python 3.10 without changing the active Python 3.8 ROS shell.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# ROS PYTHONPATH contains Python 3.8 packages; do not expose them to Python 3.10.
unset PYTHONPATH PYTHONHOME
exec bash "$PROJECT_DIR/setup/run_classifier_python.sh" "$PROJECT_DIR/main_real.py" --check-model
