FROM python:3.7-slim as base

WORKDIR /src
COPY requirements.txt /src/requirements.txt
RUN pip3 install -r requirements.txt

# install as module
COPY . .
RUN pip3 install /src

# workspace for model
WORKDIR /data

WORKDIR /src
FROM base as release
CMD python3 -m modelmanager.model

FROM base as test
# "dummy" workdir to ensure not running from source dir
WORKDIR /app
# TESTING - TEMP SOLUTION
CMD python3 -m modelmanager.model --workspace=/data --routefile=/src/tests/route_json/gps_generic.json --dumpresponse="gps_scored_route.json" --datafile /src/data/txdot_cris_20170427.csv && /src/tests/check_file_existence.sh /data

FROM base as selftest
# TESTING - TEMP SOLUTION
WORKDIR /src
CMD python3 ./modelmanager/model.py --workspace="${workspace}" --routefile=./tests/route_json/gps_generic.json --dumpresponse="gps_scored_route.json" --datafile ./data/txdot_cris_20170427.csv && /src/tests/check_file_existence.sh /data
