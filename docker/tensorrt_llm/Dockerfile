FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs python-is-python3 zsh wget
RUN pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

RUN chsh -s /bin/zsh \
    && wget -O /etc/zsh/.zshrc https://git.grml.org/f/grml-etc-core/etc/zsh/zshrc

RUN groupadd -g 67761 haydnj \
    && useradd -m -u 67761 -g 67761 haydnj