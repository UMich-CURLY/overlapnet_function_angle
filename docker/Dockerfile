FROM tensorflow/tensorflow:1.15.4-gpu-py3

USER root

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

RUN apt-get update \
    && apt-get install -y python3-pip python3-tk \
    && apt-get install -y vim \
    && apt-get install -y git

RUN pip install --upgrade pip

RUN pip install setuptools==41.0.0

RUN pip install keras==2.1.5

RUN pip install sklearn

RUN pip install matplotlib==3.0.3

RUN pip install tqdm

USER tensorflow




