FROM python:3.7-bullseye
#FROM python:3.8-buster
#FROM python:3.10-buster

WORKDIR /src
COPY . .

RUN pip3 install -r requirements.txt

CMD python3 ./code/model.py
