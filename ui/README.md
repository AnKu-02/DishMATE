# Workcell web UI

A local web app for running robot use cases with live camera feedback. Only the
dishwashing use case exists so far.

## Run

In Ubuntu-20.04, from the project folder, with the robot in Automatic mode:

```bash
cd /mnt/c/Users/aimat/Desktop/Dishwashing_Maira
source ./activate_wsl.sh
python -m ui.backend.server
```

Then open <http://localhost:8080> in a browser on this computer and choose
**Dishwashing**. Ctrl+C in the terminal stops the server; if the robot is running,
it first finishes the plate in the gripper.

## Dishwashing page

- **Live camera** from the ROS camera topic in `settings.json`.
- **Start** runs one plate per pick point. Each plate: observe pose (open gripper,
  photo, classify) → pick (close, wait 2 s) → lift → via → drop (open, wait 5 s) →
  via → observe pose. Every plate goes to the same drop point; nothing is sorted yet.
- **Pause** holds the robot before its next move; **Resume** continues.
- **Stop** while running finishes the plate in the gripper and returns to the observe
  pose. Stop before the grasp ends without picking. Stop while paused ends at once and
  leaves the robot where it is. The pendant emergency stop is still the only immediate stop.
- **Latest plate, counts and history** show each result and its dirty score (above 0.5
  means dirty). Photos are saved under `captures/` and logged in `captures/results.csv`.

Robot points come from `sort_points` in `settings.json`, the same defaults
`pick_cycles.py` uses. The robot connection is made when you press Start, so the page
and camera work while the robot is off. Robot errors (for example Manual mode) appear
on the page.

## Layout

- `backend/server.py`: web server (localhost only), camera stream and API.
- `backend/dishwashing.py`: the dishwashing run: sequence, pause/stop, results.
- `backend/camera.py`: latest ROS camera frame, JPEG stream, fresh photos.
- `frontend/`: `index.html`, `style.css`, `app.js` (no build step, no internet needed).

To add a use case, add an entry to `USE_CASES` and a session in `server.py`, and a view
in the frontend. Tests: `python -m unittest tests.test_ui_dishwashing`.
