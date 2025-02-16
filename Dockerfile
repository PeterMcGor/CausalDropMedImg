FROM nvcr.io/nvidia/pytorch:24.01-py3

# Saturn Cloud environment setup
ENV CONDA_DIR=/opt/saturncloud
ENV CONDA_BIN=${CONDA_DIR}/bin
ENV NB_USER=jovyan
ENV NB_UID=1000
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

# Install system dependencies including graphviz
RUN apt-get update && apt-get install -y \
    wget \
    git \
    sudo \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Create Saturn Cloud user
RUN useradd -m -s /bin/bash -N -u ${NB_UID} ${NB_USER} && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/notebook

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh

# Create Python environment
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
RUN chown -R ${NB_UID}:${NB_UID} ${CONDA_DIR} && \
    chown -R ${NB_UID}:${NB_UID} /opt/nnunet_resources

# Set PATH
ENV PATH=/opt/saturncloud/envs/saturn/bin:/opt/saturncloud/bin:${PATH}

USER ${NB_USER}
WORKDIR ${HOME}

RUN echo '' > ${CONDA_DIR}/conda-meta/history
