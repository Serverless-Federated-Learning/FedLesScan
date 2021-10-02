#!/usr/bin/env bash
# shellcheck disable=SC2002

pip install snakeviz

python run_handlers.py --out mnist-out-long --preload-dataset \
  --batch-size 10 --epochs 5 \
  --dataset "mnist" 2>&1 | tee tmp.txt
  echo "Results for client running long training"
cat tmp.txt | grep "Avg. duration" | cat

snakeviz -p 8084 mnist-out-short-cold.prof &
snakeviz -p 8085 mnist-out-short-warm.prof &
wait

#python run_handlers.py --out mnist-out.prof --preload-dataset --dataset "mnist"
#snakeviz mnist-out.prof
