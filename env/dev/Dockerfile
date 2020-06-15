ARG base_from=ubuntu:18.04
FROM ${base_from}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl \
    git \
    shellcheck

ARG python_version=3.8.2

ENV HOME /root
ENV PATH $HOME/.pyenv/bin:$PATH
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv \
    && echo 'eval "$(pyenv init -)"' >> ~/.bashrc \
    && eval "$(pyenv init -)" \
    && pyenv install -v ${python_version} \
    && pyenv global ${python_version} \
    && pip install pre-commit

# https://github.com/nodesource/distributions
ARG nodejs_version=14
RUN curl -sL https://deb.nodesource.com/setup_${nodejs_version}.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs

# Install shfmt
# https://github.com/mvdan/sh
# Need to install golang
RUN curl https://dl.google.com/go/go1.14.4.linux-amd64.tar.gz > go1.14.4.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.14.4.linux-amd64.tar.gz \
    && GO111MODULE=on /usr/local/go/bin/go get mvdan.cc/sh/v3/cmd/shfmt \
    && cp ~/go/bin/shfmt /usr/local/bin/shfmt \
    && rm go1.14.4.linux-amd64.tar.gz \
    && rm -rf /go \
    && rm -rf /usr/local/go/

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
