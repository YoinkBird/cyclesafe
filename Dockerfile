FROM python:3.7-bullseye as base

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

# data storage
# configure pseudo-IPC data-passing directories via symlinks
# ./output/ : model  reads/writes here
# hack - for now just make whatever changes necessary to dockerfile
RUN rm -rf /app/output
WORKDIR /data

WORKDIR /app
FROM base as release
CMD python3 ./code/model.py

FROM base as test
# TESTING - TEMP SOLUTION
#CMD ln -sv /app/t/route_json/gps_generic.json /data/gps_input_route.json && python3 ./code/model.py
#CMD python3 ./code/model.py --routefile=t/route_json/gps_generic.json
# override workspace, override path to input route file
#CMD python3 ./code/model.py --workspace=/data --routefile=t/route_json/gps_generic.json
# FAIL plan: override workspace, rely on default path to input route file
#CMD ln -sv /app/t/route_json/gps_generic.json /data/gps_input_route.json && python3 ./code/model.py --workspace=/data
# pass: plan: override workspace, override path to input route file from workspace
CMD cp /app/t/route_json/gps_generic.json /data/gps_input_route.json && python3 ./code/model.py --workspace=/data --routefile=/data/gps_input_route.json ; set -ex;  ls -ltr /data | grep "gps_input_route.json"; ls -ltr /data | grep  "gps_scored_route.json"; ls -ltr /data | grep  "human_read_dectree.pkl"; echo "PASS"
