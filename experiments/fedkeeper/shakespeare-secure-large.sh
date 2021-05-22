#!/usr/bin/env bash

python3 fedless-shakespeare.py --config config-indep-gcloud-shakespeare.yaml \
  --n-clients 355 \
  --clients-per-round 100 \
  --allowed-stragglers 50 \
  --accuracy-threshold 0.5
