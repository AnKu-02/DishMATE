#!/usr/bin/env bash
# Read-only SDK diagnostics; does not initialize robot motion.
set -e
grep -E 'ROS_MASTER_URI|ROS_IP|ROS_HOSTNAME' /root/.bash_neura_aliases || true
source /opt/ros/noetic/setup.bash
/root/neura/api_ws/venv/bin/python -c 'import rospy, cv_bridge, cv2; print("ROS camera dependencies available")'
