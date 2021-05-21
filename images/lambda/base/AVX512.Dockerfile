FROM public.ecr.aws/lambda/python:3.8

# Install tensorflow beforehand to improve docker build/push times
RUN pip install tensorflow==2.4

RUN yum install -y git

# Install fedless from built wheel
COPY ./dist/fedless*.whl .
RUN pip install fedless*.whl

RUN pip uninstall -y tensorflow
RUN pip install intel-tensorflow-avx512==2.4.0
