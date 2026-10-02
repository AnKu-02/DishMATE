# Source from Ubuntu-20.04: source ./activate_wsl.sh
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo 'Use: source ./activate_wsl.sh'
    exit 1
fi
_dish_env=/root/.venvs/dishwashing-maira
if [ ! -f "$_dish_env/bin/activate" ]; then
    echo "Environment missing: $_dish_env"
    unset _dish_env
    return 1
fi
if command -v deactivate >/dev/null 2>&1; then deactivate; fi
source /opt/ros/noetic/setup.bash
if [ -f /root/neura/api_ws/devel/setup.bash ]; then
    source /root/neura/api_ws/devel/setup.bash
fi
_dish_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eval "$("$_dish_env/bin/python" "$_dish_project/settings.py")"
unset _dish_project
source "$_dish_env/bin/activate"
unset _dish_env
