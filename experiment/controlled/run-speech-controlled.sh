#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=270
clients_per_round=100
allowed_stragglers=82
accuracy_threshold=0.9
rounds=25
straggler_percent=0.2

# n_clients=10
# clients_per_round=3
# allowed_stragglers=4
# accuracy_threshold=0.9
# rounds=7

# straggler_percent=0.3

# shellcheck disable=SC2034
# for curr_repeat in {1..1}; do


python -m fedless.core.scripts \
  -d "speech" \
  -s "fedless" \
  -c "$script_dir/speech-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/controlled/speech-d-"$straggler_percent"" \
  --rounds "$rounds" \
  --timeout 90000 \
  --mock \
  --simulate-stragglers "$straggler_percent"

python -m fedless.core.scripts \
  -d "speech" \
  -s "fedless_enhanced" \
  -c "$script_dir/speech-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/controlled/speech-enhanced-d1-"$straggler_percent"" \
  --rounds "$rounds" \
  --timeout 90000 \
  --mock \
  --simulate-stragglers "$straggler_percent"

# python -m fedless.core.scripts \
#   -d "speech" \
#   -s "fedprox" \
#   -c "$script_dir/speech-demo.yaml" \
#   --clients "$n_clients" \
#   --clients-in-round "$clients_per_round" \
#   --stragglers "$allowed_stragglers" \
#   --max-accuracy "$accuracy_threshold" \
#   --out "$root_directory/out/controlled/speech-d-prx-0.1-"$straggler_percent"" \
#   --rounds "$rounds" \
#   --timeout 90000 \
#   --mock \
#   --simulate-stragglers "$straggler_percent" \
#   --mu 0.1 &

# done
