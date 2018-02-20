FROM tensorflow/tensorflow:1.4.0-gpu-py3

RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/berge-brain/data_augmentation