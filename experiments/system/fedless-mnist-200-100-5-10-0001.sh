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
done
#'{"session_id": "03de3380-249e-49d3-b5b3-3d8336190197", "round_id": 2, "client_id": "b6ecf358-817a-457b-8e0a-2c87fdf509fc", "database": {"host": "138.246.233.207", "port": 31532, "username": "my-admin-user-623", "password": "$c3*8UxuBTeLGSxpfJE8^9n*BYg&ch&v"}, "http_headers": null}'
