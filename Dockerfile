FROM python:3.7-slim as base

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY . .


WORKDIR /app
FROM base as release
CMD python3 ./code/model.py

FROM base as test
# TESTING - TEMP SOLUTION
CMD ln -sv ../t/route_json/gps_generic.json output/gps_input_route.json && python3 ./code/model.py && bash ./t/check_file_presence.sh output/
