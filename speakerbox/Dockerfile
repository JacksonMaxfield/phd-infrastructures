FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install updates and OS deps
RUN apt update -y
RUN apt install -y \
    git \
    software-properties-common \
    libsndfile1 \
    ffmpeg \
    curl

# Install and setup Python and pip
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set python3.10 to default python3 and python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set pip3 to default pip
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Update and upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Pull repo
RUN git clone https://github.com/JacksonMaxfield/phd-infrastructures.git

# Install shared requirements
RUN pip install -r phd-infrastructures/requirements.txt

# Move to speakerbox specific and install
WORKDIR /phd-infrastructures/speakerbox/
RUN pip install -r requirements.txt
