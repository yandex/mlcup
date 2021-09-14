RUN apt-get update -qq && apt-get install -y \
      libcurl4-openssl-dev \
      build-essential \
      apt-utils && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sLo ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /usr/conda && \
    rm ~/miniconda.sh


ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/usr/conda/bin:$PATH

RUN conda install -y -c pytorch \
      pytorch \
      tensorflow \
      torchvision \
      numpy \
      scipy \
      pandas \
      scikit-learn \
      scikit-image \
      matplotlib \
      pillow \
      tqdm \
      cython \
      nltk \
      joblib && \
    conda install -y -c huggingface \
      transformers && \
    conda install -y -c conda-forge \
      opencv \
      pytorch-lightning \
      sentencepiece \
      editdistance \
      hydra-core && \
    conda clean --all -y

RUN python3.8 -m pip install --no-cache-dir \
      opencv-python