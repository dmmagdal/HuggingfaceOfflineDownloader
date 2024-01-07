# Docker file to run a container that will use the optimum-cli to
# convert pytorch models from Huggingface to ONNX in an environment
# that is compatible with transformers.js.

# Load Python 3.9 image.
FROM python:3.9.18-slim

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /run-onnx-converter
WORKDIR /run-onnx-converter
ADD . /run-onnx-converter

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt