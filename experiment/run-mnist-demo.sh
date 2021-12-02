#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=5
clients_per_round=1
allowed_stragglers=0
accuracy_threshold=0.6
rounds=2

# shellcheck disable=SC2034
for curr_repeat in {1..1}; do
  python3 -m fedless.benchmark.scripts \
    -d "mnist" \
    -s "fedkeeper" \
    -c mnist-demo.yaml \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/fedkeeper-mnist-demo" \
    --rounds "$rounds" \
    --timeout 90 \
    --separate-invokers \
    --tum-proxy
  sleep 600
  exit 0

done
