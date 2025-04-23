FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
LABEL maintainer="James Bond"
RUN apt-get update \
    && apt-get install -y wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b
COPY . /app
WORKDIR /app
RUN conda env create -f environment.yaml
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 7860
ENV MODELSCOPE_CACHE=/app/model/modelscope
CMD conda init bash \
    && . ~/.bashrc \
    && conda activate anytext \
    && python demo.py
