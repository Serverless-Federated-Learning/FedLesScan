#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=270
clients_per_round=120
allowed_stragglers=10
accuracy_threshold=0.9
rounds=20

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
  --out "$root_directory/out/speech" \
  --rounds "$rounds" \
  --timeout 90000 \
  --mock 

sleep 100

python -m fedless.core.scripts \
  -d "speech" \
  -s "fedless_enhanced" \
  -c "$script_dir/speech-demo.yaml" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$root_directory/out/speech-enhanced" \
  --rounds "$rounds" \
  --timeout 90000 \
  --mock 
exit 0

# done
