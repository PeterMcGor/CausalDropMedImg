# Start from NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Saturn Cloud environment setup
ENV CONDA_DIR=/opt/saturncloud
ENV CONDA_BIN=${CONDA_DIR}/bin
ENV NB_USER=jovyan
ENV NB_UID=1000
ENV NB_GID=1000
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    sudo \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user and group properly for Saturn Cloud
RUN groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -m -s /bin/bash -N -u ${NB_UID} -g ${NB_GID} ${NB_USER} && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/notebook

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh

# Create Python environment with Python 3.9
RUN ${CONDA_BIN}/conda create -y -n saturn python=3.9

# Copy environment file
COPY environment.yml ./

# Install using conda environment
RUN ${CONDA_BIN}/conda env update -n saturn -f environment.yml

# Set up nnUNet paths
ARG resources="/opt/nnunet_resources"
ENV nnUNet_raw=$resources"/nnUNet_raw" \
    nnUNet_preprocessed=$resources"/nnUNet_preprocessed" \
    nnUNet_results=$resources"/nnUNet_results"

# Create nnUNet directories
RUN mkdir -p /opt/nnunet_resources/nnUNet_raw && \
    mkdir -p /opt/nnunet_resources/nnUNet_preprocessed && \
    mkdir -p /opt/nnunet_resources/nnUNet_results

# Set permissions
RUN chown -R ${NB_USER}:${NB_USER} ${CONDA_DIR} && \
    chown -R ${NB_USER}:${NB_USER} /opt/nnunet_resources && \
    chown -R ${NB_USER}:${NB_USER} ${HOME}

# Set PATH
ENV PATH=/opt/saturncloud/envs/saturn/bin:/opt/saturncloud/bin:${PATH}

USER ${NB_USER}
WORKDIR ${HOME}

RUN echo '' > ${CONDA_DIR}/conda-meta/history
