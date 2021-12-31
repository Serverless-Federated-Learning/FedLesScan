#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=5
clients_per_round=3
allowed_stragglers=2
accuracy_threshold=0.6
rounds=10

# shellcheck disable=SC2034
for curr_repeat in {1..1}; do
  python -m fedless.core.scripts \
    -d "mnist" \
    -s "fedless_mock" \
    -c mnist-demo.yaml \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/fedkeeper-mnist-demo" \
    --rounds "$rounds" \
    --timeout 90000 \
    --separate-invokers \

  sleep 600
  exit 0

done
