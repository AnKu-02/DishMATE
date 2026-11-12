#!/usr/bin/env bash
# Run in Ubuntu-20.04. Keeps system Python and the ROS environment unchanged.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLASSIFIER_ENV=/root/.venvs/dishwashing-classifier
TOOLS_ENV=/root/.local/share/dishwashing-python-tools
if [ -e "$CLASSIFIER_ENV" ]; then
    echo "Refusing to overwrite existing environment: $CLASSIFIER_ENV"
    exit 1
fi
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
if [ ! -x "$TOOLS_ENV/bin/uv" ]; then
    /usr/bin/python3 -m venv "$TOOLS_ENV"
    "$TOOLS_ENV/bin/python" -m pip install uv
fi
"$TOOLS_ENV/bin/uv" python install 3.10
"$TOOLS_ENV/bin/uv" venv --python 3.10 --seed "$CLASSIFIER_ENV"
"$TOOLS_ENV/bin/uv" pip install --python "$CLASSIFIER_ENV/bin/python" -r "$PROJECT_DIR/requirements.txt"
bash "$PROJECT_DIR/setup/enable_tensorflow_cuda.sh"
bash "$PROJECT_DIR/tests/check_classifier.sh"
