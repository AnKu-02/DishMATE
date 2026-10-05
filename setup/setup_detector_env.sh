#!/usr/bin/env bash
# Separate training environment; does not modify the ROS or TensorFlow environments.
set -euo pipefail
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
UV=/root/.local/share/dishwashing-python-tools/bin/uv
ENV_DIR=/root/.venvs/dishwashing-detector
if [ ! -x "$ENV_DIR/bin/python" ]; then
    "$UV" venv --python 3.10 --seed "$ENV_DIR"
fi
"$UV" pip install --python "$ENV_DIR/bin/python" torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$ENV_DIR/bin/python" ultralytics==8.4.19
"$ENV_DIR/bin/python" -m pip check
