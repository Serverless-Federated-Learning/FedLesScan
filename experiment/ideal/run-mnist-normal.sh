#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=300
clients_per_round=100
allowed_stragglers=100
accuracy_threshold=0.99
rounds=80

# shellcheck disable=SC2034
# for curr_repeat in {1..1}; do
python -m fedless.core.scripts \
  -d "mnist" \
  -s "fedless_enhanced" \
  -c "$script_dir/mnist-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/ideal2/mnist-enhanced" \
  --rounds "$rounds" \
  --timeout 90000 \
  --mock 

sleep 2

python -m fedless.core.scripts \
  -d "mnist" \
  -s "fedless" \
  -c "$script_dir/mnist-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/ideal2/mnist" \
  --rounds "$rounds" \
  --timeout 90000 \
  --mock 


exit 0

# done
