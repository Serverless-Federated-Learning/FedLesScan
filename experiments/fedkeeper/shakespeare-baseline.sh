#!/usr/bin/env bash

python3 fedless-shakespeare.py --config config-indep-gcloud-shakespeare.yaml \
  --n-clients 30 \
  --clients-per-round 30 \
  --allowed-stragglers 10 \
  --accuracy-threshold 0.80
