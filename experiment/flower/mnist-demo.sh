#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=30
clients_per_round=30
allowed_stragglers=30
accuracy_threshold=0.99
rounds=2


dataset_name="mnist"
client_timeout=60
base_out_dir="$root_directory/out/mnist-flower"
config_dir="$script_dir/mnist-demo.yaml"
echo $base_out_dir


python -m fedless.core.scripts \
  -d "mnist" \
  -s "fedless" \
  -c "$config_dir" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$base_out_dir/$dataset_name" \
  --rounds "$rounds" \
  --timeout "$client_timeout" 
