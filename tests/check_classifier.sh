#!/usr/bin/env bash
# Uses Python 3.10 without changing the active Python 3.8 ROS shell.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLASSIFIER_PYTHON=/root/.venvs/dishwashing-classifier/bin/python
if [ ! -x "$CLASSIFIER_PYTHON" ]; then
    echo "Classifier environment missing: $CLASSIFIER_PYTHON" >&2
    exit 1
fi
# ROS PYTHONPATH contains Python 3.8 packages; do not expose them to Python 3.10.
unset PYTHONPATH PYTHONHOME
exec "$CLASSIFIER_PYTHON" "$PROJECT_DIR/main_real.py" --check-model
