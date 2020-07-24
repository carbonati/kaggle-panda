ARG PYVERSION="3.6.5"
ARG PYTORCH="1.5.1"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    sudo \
    git \
    unzip \
    htop \
    libglib2.0-0 \
    gnupg \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# miniconda and python
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==${PYVERSION} \
 && conda clean -ya

# instal pytorch-CUDA
RUN conda install -y -c pytorch cudatoolkit=${CUDA} \
 && conda clean -ya

# requirements and apex install
RUN pip install -r requirements.txt
RUN git clone https://github.com/NVIDIA/apex
RUN apex/python setup.py install --cuda_ext --cpp_ext

# Set the default command to python3
CMD ["python3"]
