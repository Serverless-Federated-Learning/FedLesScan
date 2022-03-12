#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=300
clients_per_round=200
allowed_stragglers=200
accuracy_threshold=0.99
rounds=60


dataset_name="mnist"
client_timeout=60
base_out_dir="$root_directory/out/real_world/linear"
config_dir="$script_dir/$dataset_name-$n_clients-$clients_per_round.yaml"
echo $base_out_dir

# shellcheck disable=SC2034
for straggler_percent in 0; do
  # python -m fedless.core.scripts \
  #   -d "mnist" \
  #   -s "fedless" \
  #   -c "$config_dir" \
  #   --clients "$n_clients" \
  #   --clients-in-round "$clients_per_round" \
  #   --stragglers "$allowed_stragglers" \
  #   --max-accuracy "$accuracy_threshold" \
  #   --out "$base_out_dir/$dataset_name-$straggler_percent" \
  #   --rounds "$rounds" \
  #   --timeout "$client_timeout" \
  #   --simulate-stragglers "$straggler_percent"

  # sleep1

  python -m fedless.core.scripts \
    -d "mnist" \
    -s "fedprox" \
    -c "$config_dir" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/$dataset_name-prox-$straggler_percent" \
    --rounds "$rounds" \
    --timeout "$client_timeout" \
    --simulate-stragglers "$straggler_percent" \
    --mu 0.001
  
  sleep 1
  
  # python -m fedless.core.scripts \
  #   -d "mnist" \
  #   -s "fedless_enhanced" \
  #   -c "$config_dir" \
  #   --clients "$n_clients" \
  #   --clients-in-round "$clients_per_round" \
  #   --stragglers "$allowed_stragglers" \
  #   --max-accuracy "$accuracy_threshold" \
  #   --out "$base_out_dir/$dataset_name-enhanced-$straggler_percent" \
  #   --rounds "$rounds" \
  #   --timeout "$client_timeout" \
  #   --simulate-stragglers "$straggler_percent"

  
done

exit 0

# done
