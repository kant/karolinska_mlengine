FROM tensorflow/tensorflow:1.4.0
RUN pip install --upgrade google-cloud-storage
RUN pip install --upgrade google-api-python-client
RUN pip install https://pypi.python.org/packages/33/49/c814d6d438b823441552198f096fcd0377fd6c88714dbed34f1d3c8c4389/google_auth_httplib2-0.0.3-py2.py3-none-any.whl#md5=3b1262076664f88dc415e604d7b91015

# Install GCLOUD
ENV CLOUD_SDK_REPO cloud-sdk-xenial
RUN echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update && apt-get install -y google-cloud-sdk
RUN apt-get install -y python-tk

RUN apt-get --allow-unauthenticated -qq install -y libsm6 libxext6 && pip install -q -U opencv-python

RUN apt-get install -y git
RUN pip install git+https://github.com/berge-brain/data_augmentation