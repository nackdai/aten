FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
    && apt-key fingerprint 0EBFCD88

RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

RUN apt-get update \
    && apt-get install -y docker-ce docker-ce-cli containerd.io

# https://github.com/nektos/act
RUN curl https://raw.githubusercontent.com/nektos/act/master/install.sh | bash

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

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
