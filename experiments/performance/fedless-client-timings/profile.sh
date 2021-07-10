#!/usr/bin/env bash

pip install snakeviz

python run_handlers.py --out femnist-out-short --preload-dataset \
  --batch-size 128 --epochs 1 \
  --dataset "femnist" 2>&1 | tee tmp.txt
echo "Results for client running short training"
cat tmp.txt | grep "Avg. duration" | cat

python run_handlers.py --out femnist-out-long --preload-dataset \
  --batch-size 10 --epochs 5 \
  --dataset "femnist" 2>&1 | tee tmp.txt
  echo "Results for client running long training"
cat tmp.txt | grep "Avg. duration" | cat

snakeviz -p 8084 femnist-out-short-cold.prof &
snakeviz -p 8085 femnist-out-short-warm.prof &
wait

#python run_handlers.py --out mnist-out.prof --preload-dataset --dataset "mnist"
#snakeviz mnist-out.prof
