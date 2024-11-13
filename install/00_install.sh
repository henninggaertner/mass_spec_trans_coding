
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
    wget \
    cmake \
    git \
    nano \
    vim \
    zsh \
    libglib2.0-0 \
    libbz2-dev \
    zlib1g-dev libffi-dev build-essential curl tcl-dev tk-dev uuid-dev lzma-dev liblzma-dev libssl-dev libsqlite3-dev # python make dependencies
    #python3-pip \
    #python-is-python3 \

# install custom python 3.7 version
RUN mkdir /opt/python3.7
RUN wget https://www.python.org/ftp/python/3.7.16/Python-3.7.16.tgz
RUN tar -xvf Python-3.7.16.tgz
RUN cd Python-3.7.16 && ./configure --prefix=/opt/python3.7 && make && make install
# cleanup
RUN rm -rf Python-3.7.16.tgz Python-3.7.16
RUN echo 'PATH="/opt/python3.7/bin:${PATH}"' > /etc/profile.d/python37.sh
RUN chmod +x /etc/profile.d/python37.sh
RUN source /etc/profile.d/python37.sh
RUN ln -s /opt/python3.7/bin/python3 /usr/bin/python

# Install requirements.txt
RUN python -m pip install -r /install/requirements.txt

# Install development tool for PyCharm for remote debugging
RUN python -m pip install pydevd-pycharm~=211.7628.24

# Clean up pip cache to save some space
RUN rm -rf /root/.cache/pip

# install oh-my-zsh
RUN git clone "https://github.com/robbyrussell/oh-my-zsh.git" "${HOME}/.oh-my-zsh"
RUN cp "${HOME}/.oh-my-zsh/templates/zshrc.zsh-template" "${HOME}/.zshrc"
