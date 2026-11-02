@echo off
rem Volumetric Texture Generator launcher (Windows).
rem Runs run.ps1 with script execution allowed for this invocation only.
rem Usage:
rem   run.cmd                     (GUI)
rem   run.cmd cli --size 64 ...   (CLI)
rem   run.cmd --gpu ...           (also install optional GPU deps)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
