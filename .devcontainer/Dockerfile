# hadolint global ignore=DL3006
ARG BASE_IMAGE
FROM $BASE_IMAGE

ENV SHELL /bin/bash

# User settings
ARG USERNAME=autoware
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Env vars & bash customization
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONPATH=/home/autoware/.local/bin:$PYTHONPATH
ENV GIT_PS1_SHOWDIRTYSTATE=1
ENV GIT_PS1_SHOWSTASHSTATE=1
ENV GIT_PS1_SHOWUNTRACKEDFILES=1
ENV GIT_PS1_SHOWUPSTREAM=verbose
ENV GIT_PS1_DESCRIBE_STYLE=contains
ENV GIT_PS1_SHOWCOLORHINTS=1
ENV PROMPT_COMMAND="__git_ps1 '"'${VIRTUAL_ENV:+($(basename "$VIRTUAL_ENV")) }'"\[\033[01;33m\](container) \[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]' '$ '"

# Install packages via apt
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
  bash-completion \
  gosu \
  git \
  htop \
  vim \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* "$HOME"/.cache

# Install packages via pip
# RUN python3 -m pip install --no-cache-dir --root / \
#   todo

# Add user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME
