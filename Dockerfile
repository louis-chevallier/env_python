#FROM linuxmintd/mint20-amd64
#FROM python:3.8
#FROM continuumio/miniconda3
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV TZ=Europe/Paris

# set the working directory in the container
WORKDIR /tools
RUN apt-get update --fix-missing && DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends build-essential git make wget
RUN apt-get install  --assume-yes libgl1-mesa-glx

COPY  buildenv.sh .
RUN pwd
RUN bash -ic 'source buildenv.sh && build_python_env_docker install  2>&1 | tee trace_building_env.txt'

WORKDIR /data
# origin of folder data = https://drive.google.com/drive/folders/1pFntQq6AovwZKt3L5BPM3X9NG6XdxCtw?usp=sharing ( google drive)  
# can be downloaded by python module gdown
# gdown --fuzzy https://drive.google.com/drive/folders/1pFntQq6AovwZKt3L5BPM3X9NG6XdxCtw?usp=sharing
# then gunzip runtime_data.tar.gz
# then tar xf runtime_data.tar


RUN pwd
COPY data .
RUN ls

WORKDIR /code
#RUN git clone -b nosegfault https://louis-chevallier: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@github.com/louis-chevallier/cara.git
#WORKDIR /code/cara

COPY  server.py .
EXPOSE 8080
WORKDIR /code
CMD [ "python", "server.py" ] 
