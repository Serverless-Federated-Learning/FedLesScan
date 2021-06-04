#!/usr/bin/env bash

# shellcheck disable=SC2043
for clients_in_round in 5; do
  python3 fedless-leaf.py --config config-indep-openwhisk-femnist.yaml \
    --dataset "femnist" \
    --n-clients 15 \
    --clients-per-round $clients_in_round \
    --allowed-stragglers $((clients_in_round / 2)) \
    --accuracy-threshold 0.9 \
    --no-aggregate-online
done
