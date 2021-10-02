#!/usr/bin/env bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=200
clients_per_round=25
allowed_stragglers=10
accuracy_threshold=0.9
max_rounds=100

# shellcheck disable=SC2034
for i in {3..3}; do
    python3 -m fedless.benchmark.scripts \
		-d "femnist" \
		-s "fedless" \
		-c fedless-femnist-200-25-5-10-0001.yaml \
		--clients "$n_clients" \
		--clients-in-round "$clients_per_round" \
		--stragglers "$allowed_stragglers" \
		--rounds "$max_rounds" \
		--max-accuracy "$accuracy_threshold" \
		--out "$root_directory/out/fedless-femnist-200-25-5-10-0001" \
		--aggregate-online \
		--no-tum-proxy \
		--timeout 120
    sleep 610
done

wait
