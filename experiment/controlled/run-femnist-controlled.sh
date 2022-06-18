#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=200
clients_per_round=100
allowed_stragglers=81
accuracy_threshold=0.9
rounds=30
straggler_percent=0.1

base_out_dir="$root_directory/out/controlled-expo"


# shellcheck disable=SC2034
# for curr_repeat in {1..1}; do
for straggler_percent in 0.1 0.2 0.3; do
  python -m fedless.controller.scripts \
    -d "femnist" \
    -s "fedlesscan" \
    -c "$script_dir/femnist-demo.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/femnist-enhanced-"$straggler_percent"" \
    --rounds "$rounds" \
    --aggregate-online \
    --timeout 900 \
    --mock \
    --simulate-stragglers "$straggler_percent"

  sleep 2

  python -m fedless.controller.scripts \
    -d "femnist" \
    -s "fedavg" \
    -c "$script_dir/femnist-demo.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/femnist-"$straggler_percent"" \
    --rounds "$rounds" \
    --aggregate-online \
    --timeout 900 \
    --mock \
    --simulate-stragglers "$straggler_percent"

  sleep 2
  
done
# python -m fedless.controller.scripts \
#   -d "femnist" \
#   -s "fedprox" \
#   -c "$script_dir/femnist-demo.yaml" \
#   --clients "$n_clients" \
#   --clients-in-round "$clients_per_round" \
#   --stragglers "$allowed_stragglers" \
#   --max-accuracy "$accuracy_threshold" \
#   --out "$root_directory/out/controlled/femnist-prox-"$straggler_percent"-0.001" \
#   --rounds "$rounds" \
#   --aggregate-online \
#   --timeout 900 \
#   --mock \
#   --simulate-stragglers "$straggler_percent" \
#   --mu 0.001
  # exit 0

# done
