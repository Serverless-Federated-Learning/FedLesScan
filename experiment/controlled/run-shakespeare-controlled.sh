#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=200
clients_per_round=25
allowed_stragglers=
accuracy_threshold=0.6
rounds=30
straggler_percent=0.3

# shellcheck disable=SC2034
# for curr_repeat in {1..1}; do
python -m fedless.controller.scripts \
  -d "shakespeare" \
  -s "fedavg" \
  -c "$script_dir/shakespeare-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/shakespeare" \
  --rounds "$rounds" \
  --aggregate-online \
  --timeout 600 \
  --mock

#   sleep 600
#   exit 0

# done

python -m fedless.controller.scripts \
  -d "shakespeare" \
  -s "fedlesscan" \
  -c "$script_dir/shakespeare-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/shakespeare-enhanced" \
  --rounds "$rounds" \
  --aggregate-online \
  --timeout 600 \
  --mock