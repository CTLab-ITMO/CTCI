FROM ubuntu:latest
FROM python:3.10

WORKDIR /app
COPY ./src /app/src
COPY ./tests /app/tests
COPY ./data/split /app/data/split
COPY ./requirements.txt /app
COPY ./setup.py /app

RUN apt-get update
RUN pip install -r requirements.txt


LABEL authors="PC"

ENTRYPOINT ["top", "-b"]
