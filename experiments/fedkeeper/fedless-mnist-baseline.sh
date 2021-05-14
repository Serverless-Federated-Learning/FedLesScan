#!/usr/bin/env bash

python3 fedless-mnist-baseline.py --config fedless-config.yaml \
  --n-clients 100 \
  --clients-per-round 100 \
  --allowed-stragglers 10 \
  --accuracy-threshold 0.99
