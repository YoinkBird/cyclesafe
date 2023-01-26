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
CMD python3 ./code/model.py --resource_dir=/data --routefile=t/route_json/gps_generic.json
