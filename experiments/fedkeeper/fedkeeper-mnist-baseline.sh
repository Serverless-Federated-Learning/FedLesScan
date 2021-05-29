#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=100
clients_per_round=25
allowed_stragglers=5
accuracy_threshold=0.99

python3 fedkeeper-mnist-baseline.py \
    --config config-non-ldp.yaml \
    --n-clients "$n_clients" \
    --clients-per-round "$clients_per_round" \
    --allowed-stragglers "$allowed_stragglers" \
    --accuracy-threshold "$accuracy_threshold" \
    --log-dir "$root_directory/out/fedkeeper_mnist_${n_clients}_${clients_per_round}_${allowed_stragglers}_${accuracy_threshold}_non_ldp"

sleep 600

python3 fedkeeper-mnist-baseline.py \
    --config config-ldp.yaml \
    --n-clients "$n_clients" \
    --clients-per-round "$clients_per_round" \
    --allowed-stragglers "$allowed_stragglers" \
    --accuracy-threshold "$accuracy_threshold" \
    --log-dir "$root_directory/out/fedkeeper_mnist_${n_clients}_${clients_per_round}_${allowed_stragglers}_${accuracy_threshold}_ldp"

sleep 600

for clients_per_round in 10 25 50 75 100; do
  log_dir="$root_directory/out/fedkeeper_mnist_${n_clients}_${clients_per_round}_${allowed_stragglers}_${accuracy_threshold}"
  python3 fedkeeper-mnist-baseline.py \
    --config config.yaml \
    --n-clients "$n_clients" \
    --clients-per-round "$clients_per_round" \
    --allowed-stragglers "$allowed_stragglers" \
    --accuracy-threshold "$accuracy_threshold" \
    --log-dir "$log_dir"

  sleep 600
done



