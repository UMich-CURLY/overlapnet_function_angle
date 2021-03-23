FROM tensorflow/tensorflow:1.15.4-gpu-py3

RUN pip install --upgrade pip

RUN pip install setuptools==41.0.0

RUN pip install keras==2.1.5

RUN pip install -U sklearn

RUN pip install -U matplotlib==3.0.3

RUN pip install tqdm

USER root
RUN apt-get update \
    && apt-get install -y vim \
    && apt-get install -y git

