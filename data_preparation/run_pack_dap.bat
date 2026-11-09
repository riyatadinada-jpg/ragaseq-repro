@echo off
cd /d "%~dp0"
echo Extracting pitch from the DAP-Lab master files...
where py >nul 2>nul && (py -3 pack_dap.py & goto end)
where python >nul 2>nul && (python pack_dap.py & goto end)
echo NO_PYTHON_FOUND > dap_log.txt
pause
:end
