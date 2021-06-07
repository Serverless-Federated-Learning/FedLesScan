#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=100
clients_per_round=5
allowed_stragglers=2
accuracy_threshold=0.99

# shellcheck disable=SC2034
for curr_repeat in {1..3}; do
  python3 -m fedless.benchmark.scripts \
    -d "mnist" \
    -s "fedkeeper" \
    -c config.yaml \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/fedkeeper-mnist-baseline"
  sleep 600

  python3 -m fedless.benchmark.scripts \
    -d "mnist" \
    -s "fedkeeper" \
    -c config-fedkeeper-ldp.yaml \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/fedkeeper-mnist-ldp"
  sleep 600

done
