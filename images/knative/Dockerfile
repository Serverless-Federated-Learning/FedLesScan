FROM python:3.7-slim

# Allow statements and log messages to immediately appear in the Knative logs
RUN apt update && apt -y install git
ENV PYTHONUNBUFFERED True
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir Flask gunicorn

ENV APP_HOME ./functions/knative/client-indep
WORKDIR $APP_HOME
COPY ./functions/knative/client-indep ./

COPY ./dist/fedless*.whl .
RUN pip install --no-cache-dir fedless*.whl

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 app:app