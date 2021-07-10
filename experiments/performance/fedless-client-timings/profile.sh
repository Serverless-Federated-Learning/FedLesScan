#!/usr/bin/env bash

pip install snakeviz

python run_handlers.py --out femnist-out.prof --preload-dataset --dataset "femnist" --caching
snakeviz femnist-out.prof

#python run_handlers.py --out mnist-out.prof --preload-dataset --dataset "mnist"
#snakeviz mnist-out.prof
