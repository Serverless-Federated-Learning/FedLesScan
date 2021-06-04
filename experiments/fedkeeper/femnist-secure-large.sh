#!/usr/bin/env bash

# shellcheck disable=SC2043
for clients_in_round in 10; do
  python3 fedless-leaf.py --config config-indep-gcloud-secure-femnist.yaml \
    --dataset "femnist" \
    --n-clients 300 \
    --clients-per-round $clients_in_round \
    --allowed-stragglers $((clients_in_round / 2)) \
    --accuracy-threshold 0.5 \
    --no-aggregate-online
done

for clients_in_round in 50 100 300; do
  python3 fedless-leaf.py --config config-indep-gcloud-secure-femnist.yaml \
    --dataset "femnist" \
    --n-clients 300 \
    --clients-per-round $clients_in_round \
    --allowed-stragglers $((clients_in_round / 2)) \
    --accuracy-threshold 0.5 \
    --aggregate-online
done
