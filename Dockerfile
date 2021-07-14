FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

WORKDIR /workspace

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY src src

ENTRYPOINT ["python", "src/train_lm_ref.py"]
CMD ["--help"]





