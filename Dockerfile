#FROM tensorflow/tensorflow:2.1.0-gpu-py3
FROM tensorflow/tensorflow:latest-gpu
USER root

RUN mkdir -p /workspaces/src
RUN mkdir -p /workspaces/img
COPY requirements.txt /workspaces/src
WORKDIR /workspaces/src

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev pip wget unzip graphviz
RUN pip install --upgrade pip setuptools

RUN pip3 install -r ./requirements.txt