set -ev
# prepare
## clone into dir with model as a subdir. this keeps all filepaths relative
# cd <parentProjDir> && git clone ... server
# manually verify that 'server' is in the .gitignore of the parent project

curdir=$(dirname $0)
cd $curdir

## files from model:
serve_this=./report.md
if [ ! -r  $serve_this ]; then
  exit
fi

# startup - use python2, ran into encoding errors when converting to python3 after 2to3
# if 'port already in use', could just be from re-running
port=6419
procname="grip"
$procname --quiet $serve_this &
server_pid=$!
echo $?
# if server already running, the new PID just gets confusing
#echo $! >> server_pid.txt

# let it spin up
sleep 5

# how to kill the server
lsof -i :${port} | tee -a server_pid_lsof_${procname}.txt

#-------------------------------------------------------------------------------- 
# mock client map-ui - upload json
#+ src: https://stackoverflow.com/a/7173011
## curl --header  http://localhost:${port} --data @../t/route_json/gps_generic.json

#-------------------------------------------------------------------------------- 
# mock client map-ui - retrieve json
# xterm -hold -e curl http://localhost:${port}
curl -s http://localhost:${port} 1>/dev/null # '2>&1' causes RC of 3
if [[ $? -ne 0 ]]; then
  exit
fi


#-------------------------------------------------------------------------------- 
# mock client map-ui - view json in browser
chromium-browser --new-window --incognito http://localhost:${port}
# how about some other things?


#-------------------------------------------------------------------------------- 
# show any running servers
#cat server_pid.txt
server_pid_file="server_pid_lsof_${procname}.txt"
cat server_pid_lsof_${procname}.txt
