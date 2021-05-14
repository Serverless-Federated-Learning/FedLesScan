#!/usr/bin/env bash

docker build . -t "flower-client-test"

docker run --rm -it --network host flower-client-test