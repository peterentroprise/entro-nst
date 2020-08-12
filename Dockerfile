FROM tiangolo/uvicorn-gunicorn-machine-learning:python3.7

ENV TIMEOUT 1000

ENV GRACEFUL_TIMEOUT 1000

ENV PORT 8080

# fastAPI

RUN conda install -c conda-forge fastapi

RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

#gcloud storage

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

COPY ./service-account.json /app

# COPY ./gunicorn_config.py /app

ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"

RUN gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

# RUN gsutil -m cp -R gs://entro-haystack-models/sentence_bert /app

COPY ./app /app