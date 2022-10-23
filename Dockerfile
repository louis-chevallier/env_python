#FROM linuxmintd/mint20-amd64
#FROM python:3.8
#FROM continuumio/miniconda3
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV TZ=Europe/Paris

# set the working directory in the container
WORKDIR /code
RUN apt-get update --fix-missing && DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends build-essential git make wget
RUN apt-get install  --assume-yes libgl1-mesa-glx
RUN pwd

WORKDIR /code/
RUN pwd

RUN bash -ic 'source buildenv.sh && build_python_env_docker install  2>&1 | tee trace_building_env.txt'
WORKDIR /data
RUN gdown https://drive.google.com/drive/folders/1pFntQq6AovwZKt3L5BPM3X9NG6XdxCtw?usp=sharing
RUN pwd 
#COPY  datadir/models .
EXPOSE 8080
WORKDIR /code
CMD [ "make", "server" ] 
