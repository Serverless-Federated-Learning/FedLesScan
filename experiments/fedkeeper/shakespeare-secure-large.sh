#!/usr/bin/env bash

for clients_in_round in 50 100 300; do
  python3 fedless-leaf.py --config config-indep-gcloud-secure-shakespeare.yaml \
    --dataset "shakespeare" \
    --n-clients 300 \
    --clients-per-round $clients_in_round \
    --allowed-stragglers $((clients_in_round / 2)) \
    --accuracy-threshold 0.5
done
