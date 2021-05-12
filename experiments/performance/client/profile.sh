#!/usr/bin/env bash
set -e

pip install snakeviz
#python profile_client_handler.py --out mnist-no-dp.prof --no-ldp --runs 1 --local-epochs 5 --dataset mnist
#snakeviz mnist-no-dp.prof


python profile_client_handler.py --out shakespeare-no-dp.prof --no-ldp --runs 1 --local-epochs 1 --dataset shakespeare
python profile_client_handler.py --out shakespeare-dp.prof --ldp --runs 1 --local-epochs 1 --dataset shakespeare

snakeviz shakespeare-dp.prof


