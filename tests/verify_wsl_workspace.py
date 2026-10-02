"""Manual WSL startup and environment checks. No robot clients are created."""
import os
import subprocess

env = dict(os.environ)
for key in list(env):
    if key.startswith(('ROS_', 'VIRTUAL_ENV', 'PYTHONPATH', 'CMAKE_PREFIX_PATH')):
        env.pop(key)
env['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
env.pop('SHLVL', None)
env.pop('TERM_PROGRAM', None)

def shell(command):
    return subprocess.run(['bash', '--login', '-ic', command], env=env,
                          cwd='/mnt/c/Users/aimat/Desktop/Kanak',
                          text=True, capture_output=True, check=True).stdout

result = shell('printf "DIRECTORY=%s\\nENV=%s\\nROS=%s\\n" "$PWD" "${VIRTUAL_ENV:-none}" "${ROS_MASTER_URI:-none}"')
print(result)
assert 'DIRECTORY=/root\n' in result
assert 'ENV=none\n' in result
assert 'ROS=none\n' in result
result = shell('kanak; printf "DIRECTORY=%s\\nENV=%s\\n" "$PWD" "$VIRTUAL_ENV"')
print(result)
assert 'DIRECTORY=/mnt/c/Users/aimat/Desktop/Kanak\n' in result
assert 'ENV=/mnt/c/Users/aimat/Desktop/neura_new_sw_version/.venv\n' in result
result = shell('cd /mnt/c/Users/aimat/Desktop/Dishwashing_Maira; source ./activate_wsl.sh; python -c "import sys, rospy, cv_bridge, cv2, numpy, neurapy.robot; print(sys.executable); print(cv2.__version__, numpy.__version__)"')
print(result)
assert '/root/.venvs/dishwashing-maira/bin/python' in result
print('All startup and activation checks passed.')
