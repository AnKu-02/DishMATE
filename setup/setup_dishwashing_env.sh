#!/usr/bin/env bash
set -e
ENV_DIR=/root/.venvs/dishwashing-maira
if [ -e "$ENV_DIR" ]; then
    echo "Refusing to overwrite existing environment: $ENV_DIR"
    exit 1
fi
# ROS Noetic's compiled cv_bridge uses Ubuntu's Python 3.8 and NumPy.
# System packages are readable; pip installs stay in this dedicated venv.
/usr/bin/python3 -m venv --system-site-packages "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install \
    '/mnt/c/Users/aimat/Desktop/neura_new_sw_version/neura-sdk-v5.1.85/neura-sdk/Neurapy_v5.1.85/neurapy-5.1.85-py3-none-any.whl'
source /opt/ros/noetic/setup.bash
"$ENV_DIR/bin/python" -c 'import sys, rospy, cv_bridge, cv2, numpy, neurapy.robot; print(sys.executable); print("ROS, OpenCV, NumPy and NeuraPy imports passed")'
