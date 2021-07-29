# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.189.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT="3.8"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}
ARG HW_ACCEL="cpu"

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        liblapack-dev libblas-dev gfortran python3-opencv libnuma-dev && \
    if [ "$HW_ACCEL"="rocm" ]; then \
        wget -q -O - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add - && \
        echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/4.2/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list && \
        wget https://people.debian.org/~paravoid/python-all/unofficial-python-all.asc && \
        mv unofficial-python-all.asc /etc/apt/trusted.gpg.d && \
        echo "deb http://people.debian.org/~paravoid/python-all $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/python-all.list && \
        apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
            rocm-gdb rocm-dev rocm-dkms rocm-libs hipcub miopen-hip rccl ; \
    fi && \
    rm -rf /var/lib/apt/lists/*


COPY *.txt /tmp/pip-tmp/
RUN pip3 install gdown
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt && \
    INDEX=$([ "$HW_ACCEL"="rocm" ] && echo "https://download.pytorch.org/whl/rocm4.2/torch_stable.html" || echo "https://download.pytorch.org/whl/torch_stable.html") && \
    pip3 install --find-links $INDEX -r /tmp/pip-tmp/requirements-${HW_ACCEL}.txt && \
    rm -rf /tmp/pip-tmp
