@echo off
cd /d "%~dp0"
echo Fetching only the Saraga Hindustani pitch, tonic and metadata files for RagaSeq.
echo This can take a while. The window closes by itself when done; progress is in saraga_log.txt.
where py >nul 2>nul && (py -3 get_saraga.py & goto end)
where python >nul 2>nul && (python get_saraga.py & goto end)
echo NO_PYTHON_FOUND > saraga_log.txt
pause
:end
