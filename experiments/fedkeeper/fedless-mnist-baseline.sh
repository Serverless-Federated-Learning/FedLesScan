#!/usr/bin/env bash

python3 fedless-mnist-baseline.py --config fedless-config.yaml \
  --n-clients 100 \
  --clients-per-round 25 \
  --allowed-stragglers 5 \
  --accuracy-threshold 0.98
