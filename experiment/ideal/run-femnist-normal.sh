#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=200
clients_per_round=100
allowed_stragglers=10
accuracy_threshold=0.9
rounds=30

# shellcheck disable=SC2034
# for curr_repeat in {1..1}; do
python -m fedless.controller.scripts \
  -d "femnist" \
  -s "fedless" \
  -c "$script_dir/femnist-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/ideal2/femnist" \
  --rounds "$rounds" \
  --aggregate-online \
  --timeout 900 \
  --mock

sleep 5

python -m fedless.controller.scripts \
  -d "femnist" \
  -s "fedless_enhanced" \
  -c "$script_dir/femnist-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/ideal2/femnist-enhanced" \
  --rounds "$rounds" \
  --aggregate-online \
  --timeout 900 \
  --mock
  # exit 0

# done
