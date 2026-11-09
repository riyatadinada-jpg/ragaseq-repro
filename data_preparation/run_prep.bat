@echo off
cd /d "%~dp0"
echo Packing RagaSeq pitch data. This window closes by itself when done.
where py >nul 2>nul && (py -3 prep_ragaseq.py & goto end)
where python >nul 2>nul && (python prep_ragaseq.py & goto end)
echo NO_PYTHON_FOUND > prep_log.txt
echo Python was not found on this computer.
pause
:end
