#!/usr/bin/env bash

python3 fedkeeper-mnist-secure.py --config config-secure.yaml \
  --n-clients 50 \
  --clients-per-round 10 \
  --allowed-stragglers 2 \
  --accuracy-threshold 0.95
