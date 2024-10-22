
# Make data directory
RUN mkdir -p /data

# Necessary fix, on clean install it complained it didn't exist
RUN mkdir -p /workspace

# fix missing PUB KEY
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

export DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y install \
    build-essential \
    cmake \
    git \
    python3-pip \
    python-is-python3 \
    nano \
    vim \
    zsh \
    libglib2.0-0

# Install requirements.txt
RUN pip install -r /install/requirements.txt

# Install development tool for PyCharm for remote debugging
RUN pip install pydevd-pycharm~=211.7628.24

# Clean up pip cache to save some space
RUN rm -rf /root/.cache/pip

# install oh-my-zsh
RUN git clone "https://github.com/robbyrussell/oh-my-zsh.git" "${HOME}/.oh-my-zsh"
RUN cp "${HOME}/.oh-my-zsh/templates/zshrc.zsh-template" "${HOME}/.zshrc"
