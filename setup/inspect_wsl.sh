#!/usr/bin/env bash
set -e
for file in /root/.bash_aliases /root/.bash_profile /root/.bash_login; do
  if [ -f "$file" ]; then printf '\n%s\n' "$file"; cat "$file"; fi
done
for python in /usr/bin/python3 /mnt/c/Users/aimat/Desktop/neura_new_sw_version/.venv/bin/python /root/neura/api_ws/venv/bin/python; do
  "$python" -c 'import sys; print(sys.executable, sys.version)'
done
/mnt/c/Users/aimat/Desktop/neura_new_sw_version/.venv/bin/python -m pip show numpy tensorflow keras neurapy neurapy-ai opencv-python
ls -d /usr/bin/python3* /root/.local/share/uv/python/* 2>/dev/null || true
