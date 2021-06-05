#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=100
clients_per_round=25
allowed_stragglers=5
accuracy_threshold=0.99

# shellcheck disable=SC2034
for curr_repeat in {1..3}; do
  python3 fedkeeper-mnist-baseline.py \
    --config config-fedkeeper-non-ldp.yaml \
    --n-clients "$n_clients" \
    --clients-per-round "$clients_per_round" \
    --allowed-stragglers "$allowed_stragglers" \
    --accuracy-threshold "$accuracy_threshold" \
    --log-dir "$root_directory/out/fedkeeper_mnist_${n_clients}_${clients_per_round}_${allowed_stragglers}_${accuracy_threshold}_non_ldp"
  sleep 600

  python3 fedkeeper-mnist-baseline.py \
    --config config-fedkeeper-ldp.yaml \
    --n-clients "$n_clients" \
    --clients-per-round "$clients_per_round" \
    --allowed-stragglers "$allowed_stragglers" \
    --accuracy-threshold "$accuracy_threshold" \
    --log-dir "$root_directory/out/fedkeeper_mnist_${n_clients}_${clients_per_round}_${allowed_stragglers}_${accuracy_threshold}_ldp"
  sleep 600

done
