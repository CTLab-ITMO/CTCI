FROM ubuntu:latest
FROM python:3.10

WORKDIR /app
COPY . /app

RUN apt-get update
RUN pip install -r requirements.txt

LABEL authors="PC"

ENTRYPOINT ["top", "-b"]
