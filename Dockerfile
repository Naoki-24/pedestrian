#FROM tensorflow/tensorflow:2.1.0-gpu-py3
FROM tensorflow/tensorflow:latest-gpu
USER root

RUN mkdir -p /root/src
RUN mkdir -p /root/img
COPY requirements.txt /root/src
WORKDIR /root/src

RUN apt-get install -y libgl1-mesa-dev pip wget unzip
RUN pip install --upgrade pip setuptools

RUN pip3 install -r ./requirements.txt