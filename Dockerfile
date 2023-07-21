FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python dependencies
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Model weight files 
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=$HF_AUTH_TOKEN

ADD download.py .
ADD runner.py .
RUN python3 download.py

# App code
ADD server.py .
ADD app.py .

# Prep and start server
EXPOSE 8000

CMD python3 -u server.py
