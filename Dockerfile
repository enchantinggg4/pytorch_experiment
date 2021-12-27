FROM python:3.8-slim-buster

WORKDIR /app
COPY src .

RUN pip3 install numpy && \
    pip3 install decorator && \
    pip3 install sympy==1.4 && \
    pip3 install cffi==1.12.3 && \
    pip3 install pyyaml && \
    pip3 install pathlib2 && \
    pip3 install grpcio && \
    pip3 install grpcio-tools && \
    pip3 install protobuf && \
    pip3 install scipy && \
    pip3 install requests && \
    pip3 install attrs && \
    pip3 install Pillow && \
    pip3 install torchvision==0.2.2.post3