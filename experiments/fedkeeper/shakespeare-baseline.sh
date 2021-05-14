#!/usr/bin/env bash

python3 fedless-shakespeare.py --config fedless-config-shakespeare.yaml \
  --n-clients 40 \
  --clients-per-round 20 \
  --allowed-stragglers 10 \
  --accuracy-threshold 0.80
