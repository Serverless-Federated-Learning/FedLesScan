#!/usr/bin/env bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=200
clients_per_round=100
allowed_stragglers=10
accuracy_threshold=0.99
max_rounds=100

# shellcheck disable=SC2034
for i in {1..3}; do
    python3 -m fedless.benchmark.scripts \
		-d "mnist" \
		-s "fedless" \
		-c fedless-mnist-200-100-5-10-0001.yaml \
		--clients "$n_clients" \
		--clients-in-round "$clients_per_round" \
		--stragglers "$allowed_stragglers" \
		--rounds "$max_rounds" \
		--max-accuracy "$accuracy_threshold" \
		--out "$root_directory/out/fedless-mnist-200-100-5-10-0001" \
		--aggregate-offline \
		--no-tum-proxy \
		--timeout 60
    sleep 610
done

wait
