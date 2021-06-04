#!/usr/bin/env bash

for clients_in_round in 10; do
  python3 fedless-leaf.py --config config-indep-gcloud-secure-femnist.yaml \
    --dataset "femnist" \
    --n-clients 15 \
    --clients-per-round $clients_in_round \
    --allowed-stragglers $((clients_in_round / 2)) \
    --accuracy-threshold 0.9 \
    --no-aggregate-online
done
