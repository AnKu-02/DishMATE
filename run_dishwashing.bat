@echo off
rem Dishwashing workcell: UI backend + browser UI + RoboDK digital twin. Ctrl+C to stop.
cd /d "%~dp0"
"C:\RoboDK\Python-Embedded\python.exe" run_dishwashing.py %*
pause
