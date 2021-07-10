#!/usr/bin/env bash

pip install snakeviz

python run_handlers.py --out out.prof --preload-dataset --dataset "femnist"

snakeviz out.prof
