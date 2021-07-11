#!/usr/bin/env bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=100
clients_per_round=10
allowed_stragglers=10
accuracy_threshold=0.55
max_rounds=100

# shellcheck disable=SC2034
for curr_repeat in {1..3}; do
	python3 -m fedless.benchmark.scripts \
		-d "shakespeare" \
		-s "fedless" \
		-c fedless-shakespeare-flower-comp.yaml \
		--clients "$n_clients" \
		--clients-in-round "$clients_per_round" \
		--stragglers "$allowed_stragglers" \
		--rounds "$max_rounds" \
		--max-accuracy "$accuracy_threshold" \
		--timeout 310 \
		--out "$root_directory/out/fedless-shakespeare-flower-258-$clients_per_round" 
		#--tum-proxy --aggregate-online
	sleep 1
done
