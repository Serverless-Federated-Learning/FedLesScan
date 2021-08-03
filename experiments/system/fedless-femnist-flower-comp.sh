#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=100
clients_per_round=75
allowed_stragglers=10
accuracy_threshold=0.90
max_rounds=100

# shellcheck disable=SC2034
for curr_repeat in {1..3}; do
	# shellcheck disable=SC2043
	for clients_per_round in 75; do
		python3 -m fedless.benchmark.scripts \
			-d "femnist" \
			-s "fedless" \
			-c fedless-femnist-flower-comp.yaml \
			--clients "$n_clients" \
			--clients-in-round "$clients_per_round" \
			--stragglers "$allowed_stragglers" \
			--rounds "$max_rounds" \
			--max-accuracy "$accuracy_threshold" \
			--out "$root_directory/out/fedless-femnist-flower-$clients_per_round" \
			--timeout 60 \
			--aggregate-online
		#--tum-proxy
#		sleep 60
#		cat << EOF | mongo mongodb://my-admin-user-623:'$c3*8UxuBTeLGSxpfJE8^9n*BYg&ch&v'@138.246.233.207:31532
#use fedless
#db.dropDatabase()
#EOF
	done
done
