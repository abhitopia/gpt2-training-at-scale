FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

WORKDIR /workspace
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
