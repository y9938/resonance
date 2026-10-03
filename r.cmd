@echo off
uv run --script "%~dp0scripts\tasks.py" %*
exit /b %errorlevel%
