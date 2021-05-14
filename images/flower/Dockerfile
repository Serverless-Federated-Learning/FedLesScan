FROM python:3.8

RUN pip install flwr
RUN pip install tensorflow

COPY ./dist/fedless*.whl .
RUN pip install fedless*.whl
