FROM python:3.7-slim as base

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

# workspace for model
WORKDIR /data

WORKDIR /app
FROM base as release
CMD python3 ./modelmanager/model.py

FROM base as test
CMD ./tests/run_model_and_check_file_exisistence.sh /data
