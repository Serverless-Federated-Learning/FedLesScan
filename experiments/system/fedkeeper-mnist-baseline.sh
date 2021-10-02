#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=100
clients_per_round=25
allowed_stragglers=5
accuracy_threshold=0.99
rounds=200

# shellcheck disable=SC2034
for curr_repeat in {1..3}; do
  python3 -m fedless.benchmark.scripts \
    -d "mnist" \
    -s "fedkeeper" \
    -c fedkeeper-mnist-basic.yaml \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/fedkeeper-mnist-baseline" \
    --rounds "$rounds" \
    --timeout 90 \
    --separate-invokers \
    --tum-proxy
  sleep 600
  exit 0
  python3 -m fedless.benchmark.scripts \
    -d "mnist" \
    -s "fedkeeper" \
    -c fedkeeper-mnist-ldp.yaml \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/fedkeeper-mnist-ldp" \
    --rounds "$rounds" \
    --timeout 300 \
    --separate-invokers \
    --tum-proxy
  sleep 600

done
