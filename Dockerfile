FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

#os updates
RUN apt-get update && apt-get install libssl-dev libcurl4-openssl-dev python-dev gcc -y
RUN ln -s /run/shm /dev/shm
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

ENV PORT 8080

COPY ./app /app