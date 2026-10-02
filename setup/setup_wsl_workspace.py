"""One-time Ubuntu-20.04 shell cleanup; backups precede every edit."""
from pathlib import Path
from datetime import datetime
import shutil
import subprocess

home = Path.home()
backup = home / '.shell-backups' / ('dishwashing-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
backup.mkdir(parents=True)
paths = [home / '.bashrc', home / '.bash_aliases', home / '.profile', home / '.bash_neura_aliases']
for path in paths:
    if path.exists():
        shutil.copy2(path, backup / path.name)

rc = (home / '.bashrc').read_text()
start = rc.index('# >>> neura_new_sw_version auto-activate >>>')
end_marker = '# <<< neura_new_sw_version auto-activate <<<'
end = rc.index(end_marker, start) + len(end_marker)
rc = rc[:start] + '''# Project environments are activated explicitly.
export NEURA_NEW_SW_DIR="/mnt/c/Users/aimat/Desktop/neura_new_sw_version"
alias nsw='cd "$NEURA_NEW_SW_DIR" && source /opt/ros/noetic/setup.bash && source /root/.bash_neura_aliases && source "$NEURA_NEW_SW_DIR/.venv/bin/activate"'
alias kanak='cd /mnt/c/Users/aimat/Desktop/Kanak && source /opt/ros/noetic/setup.bash && source /root/.bash_neura_aliases && source "$NEURA_NEW_SW_DIR/.venv/bin/activate"'

# Fresh standalone WSL terminals start at the Linux home directory.
# Keep editor terminals and nested shells in their requested working directory.
if [ "${SHLVL:-1}" -le 1 ] && [ "${TERM_PROGRAM:-}" != "vscode" ]; then
    cd "$HOME"
fi
''' + rc[end:]
block = '''# Source Neura AI SDK environment
if [ -f "/root/.bash_neura_aliases" ]; then
    source "/root/.bash_neura_aliases"
fi'''
assert block in rc
rc = rc.replace(block, '# NEURA SDK setup is now explicit via nsw, kanak, or project activation.')
lines = []
for line in rc.splitlines():
    if line.startswith(('source /opt/ros/noetic/setup.bash', 'export PYTHONPATH=', 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:')):
        line = '# Disabled global project setup: ' + line
    lines.append(line)
(home / '.bashrc').write_text('\n'.join(lines) + '\n')

aliases = (home / '.bash_aliases').read_text()
cd_block = '''if [[ $- == *i* ]] && [ -d /mnt/c/Users/aimat/Desktop/Kanak ]; then
  cd /mnt/c/Users/aimat/Desktop/Kanak
fi'''
assert cd_block in aliases
aliases = aliases.replace(cd_block, '# Use the kanak alias to enter the project and activate its existing environment.')
lines = []
for line in aliases.splitlines():
    if line.startswith(('source /opt/ros/', 'export ROS_')):
        line = '# Disabled global project setup: ' + line
    lines.append(line)
(home / '.bash_aliases').write_text('\n'.join(lines) + '\n')
subprocess.run(['bash', '-n', str(home / '.bashrc')], check=True)
subprocess.run(['bash', '-n', str(home / '.bash_aliases')], check=True)
print('Startup backup:', backup)
print('Existing project files and environments were not edited.')
