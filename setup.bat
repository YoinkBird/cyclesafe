REM chdir code
start jupyter qtconsole
REM chdir ..\..
REM avoid starting duplicate servers
start jupyter notebook --port=8888 --port-retries=0


