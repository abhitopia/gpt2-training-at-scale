ARG PTH_VERSION=1.9.0-cuda11.1-cudnn8

# 1/Building apex with pytorch:*-devel
FROM pytorch/pytorch:${PTH_VERSION}-devel AS apex-builder

ARG ARG_TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0"
ENV TORCH_CUDA_ARCH_LIST=$ARG_TORCH_CUDA_ARCH_LIST

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends git && \
     rm -rf /var/lib/apt/lists/*

# Build apex
RUN echo "Setup NVIDIA Apex" && \
    tmp_apex_path="/tmp/apex" && \
    rm -rf $tmp_apex_path && \
    git clone https://github.com/NVIDIA/apex $tmp_apex_path && \
    cd $tmp_apex_path && \
    pip wheel --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# Build runtime image
FROM pytorch/pytorch:${PTH_VERSION}-runtime

# Apex
COPY --from=apex-builder /tmp/apex/apex-*.whl /tmp/apex/
RUN pip install --no-cache-dir /tmp/apex/apex-*.whl && \
    rm -fr /tmp/apex

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install torchelastic
Copy .git .git
COPY src src

ENV PYTHONPATH "${PYTHONPATH}:/workspace"
ENTRYPOINT ["python", "src/train_lm.py"]
CMD ["--help"]





