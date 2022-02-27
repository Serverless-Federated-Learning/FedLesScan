#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$script_dir")"
echo $script_dir
echo $root_directory
n_clients=270
clients_per_round=120
allowed_stragglers=10
accuracy_threshold=0.72
rounds=50

# shellcheck disable=SC2034
for curr_repeat in {1..1}; do
  python -m fedless.core.scripts \
    -d "speech" \
    -s "fedless" \
    -c "$script_dir/speech-demo.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/speech-demo" \
    --rounds "$rounds" \
    --timeout 90000 \
    --separate-invokers \
    -- mock

  sleep 600
  exit 0

done
