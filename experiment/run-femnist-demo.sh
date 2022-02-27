#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$script_dir")"
echo $script_dir
echo $root_directory
n_clients=200
clients_per_round=100
allowed_stragglers=10
accuracy_threshold=0.9
rounds=100

# shellcheck disable=SC2034
for curr_repeat in {1..1}; do
  python -m fedless.core.scripts \
    -d "femnist" \
    -s "fedless" \
    -c "$script_dir/femnist-demo.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/femnist-demo" \
    --rounds "$rounds" \
    --aggregate-online \
  	--timeout 120
    -- mock

  sleep 600
  exit 0

done
