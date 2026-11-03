@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-v23-installed.ps1" %*
if errorlevel 1 pause
