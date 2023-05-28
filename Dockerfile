# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# RUN pip3 install pytorch-lightning==1.9.1

# Create app directory
RUN mkdir -p /app
WORKDIR /app

RUN pip3 install pip==23.1.2

# Install python requirements
ARG FURY_TOKEN
COPY requirements.txt /app
RUN pip3 install -r requirements.txt --extra-index-url https://${FURY_TOKEN}:@pypi.fury.io/taranis/

#copy app
COPY . /app
