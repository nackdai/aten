ARG base_from=ubuntu:20.04
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
    && echo 'eval "$(pyenv init --path)"' >> ~/.bashrc \
    && eval "$(pyenv init --path)" \
    && pyenv install -v ${python_version} \
    && pyenv global ${python_version} \
    && pip install pre-commit

# https://github.com/nodesource/distributions
# https://github.com/nodesource/distributions/issues/1266
ARG npm_version=8.4.1
RUN apt-get update && apt-get install -y ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_current.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && npm install -g npm@${npm_version}
# https://stackoverflow.com/questions/69692842/error-message-error0308010cdigital-envelope-routinesunsupported
ENV NODE_OPTIONS --openssl-legacy-provider

# Install shfmt
# https://github.com/mvdan/sh
# Need to install golang
ARG go_version=1.17.1
RUN curl https://dl.google.com/go/go${go_version}.linux-amd64.tar.gz > go.tar.gz \
    && tar -C /usr/local -xzf go.tar.gz \
    && GO111MODULE=on /usr/local/go/bin/go install mvdan.cc/sh/v3/cmd/shfmt@latest \
    && cp ~/go/bin/shfmt /usr/local/bin/shfmt \
    && rm go.tar.gz \
    && rm -rf /go \
    && rm -rf /usr/local/go/

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
