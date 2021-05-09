#!/usr/bin/env bash

python3 fedkeeper-mnist-baseline.py --config config.yaml \
  --n-clients 100 \
  --clients-per-round 25 \
  --allowed-stragglers 5 \
  --accuracy-threshold 0.98
