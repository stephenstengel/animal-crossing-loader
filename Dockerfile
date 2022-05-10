FROM tensorflow/tensorflow:latest-gpu
RUN yes | apt-get install wget
RUN yes | apt-get install wget2
RUN pip install scikit-image matplotlib tqdm