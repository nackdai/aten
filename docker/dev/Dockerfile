ARG from=ubuntu:20.04
FROM ${from}

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
    shellcheck \
    python3 \
    python3-pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install pre-commit

# Install shfmt
# https://github.com/mvdan/sh
# Need to install golang
ARG go_version=1.24.3
RUN curl -L https://go.dev/dl/go${go_version}.linux-amd64.tar.gz -o go.tar.gz \
    && tar -C /usr/local -xzf go.tar.gz \
    && GO111MODULE=on /usr/local/go/bin/go install mvdan.cc/sh/v3/cmd/shfmt@latest \
    && cp ~/go/bin/shfmt /usr/local/bin/shfmt \
    && rm go.tar.gz \
    && rm -rf /go \
    && rm -rf /usr/local/go/
