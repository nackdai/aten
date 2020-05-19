# https://qiita.com/sugiyasu-qr/items/85a1bedb6458d4573407#%E6%96%B9%E6%B3%95-%E3%83%9B%E3%82%B9%E3%83%88%E3%83%9E%E3%82%B7%E3%83%B3%E4%B8%8A%E3%81%AE-docker-daemon-%E3%82%92%E5%85%B1%E6%9C%89%E3%81%99%E3%82%8B
# https://esakat.github.io/esakat-blog/posts/docker-in-docker/

# /var/run/docker.sock:/var/run/docker.sock

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

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash \
    && apt-get install -y nodejs

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
