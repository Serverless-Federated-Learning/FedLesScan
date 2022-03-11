#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=300
clients_per_round=200
allowed_stragglers=200
accuracy_threshold=0.99
rounds=100

base_out_dir="$root_directory/out/real_world/linear"

# shellcheck disable=SC2034
for straggler_percent in 0.7 0.5 0.3 0.1; do
  python -m fedless.core.scripts \
    -d "mnist" \
    -s "fedless_enhanced" \
    -c "$script_dir/mnist-300-100.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/mnist-enhanced" \
    --rounds "$rounds" \
    --timeout 600 \
    --simulate-stragglers "$straggler_percent"

  sleep 1

  python -m fedless.core.scripts \
    -d "mnist" \
    -s "fedless" \
    -c "$script_dir/mnist-300-100.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/mnist" \
    --rounds "$rounds" \
    --timeout 600 \
    --simulate-stragglers "$straggler_percent"

  python -m fedless.core.scripts \
    -d "mnist" \
    -s "fedprox" \
    -c "$script_dir/mnist-300-100.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/mnist-prox" \
    --rounds "$rounds" \
    --timeout 600 \
    --simulate-stragglers "$straggler_percent" \
    --mu 0.001
done

exit 0

# done
