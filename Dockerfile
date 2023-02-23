FROM python:3.7-slim as base

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY . .


WORKDIR /app
FROM base as release
CMD python3 ./modelmanager/model.py

FROM base as test
# TESTING - TEMP SOLUTION
CMD ln -sv ../tests/route_json/gps_generic.json output/gps_input_route.json && python3 ./modelmanager/model.py && bash ./tests/check_file_presence.sh output/
