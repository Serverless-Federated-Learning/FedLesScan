#!/usr/bin/env bash

for l2_norm_clip in 0.5 1.0 5.0 10.0 20.0; do
  for noise_multiplier in 0.5 1.0 2.0; do
    python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs 150 --local-epochs 10 \
      --local-batch-size 10 --clients-per-round 25 \
      --local-dp --l2-norm-clip "$l2_norm_clip" --noise-multiplier "$noise_multiplier"
  done
done

for local_epochs in 5 10; do
  python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs 100 --local-epochs "$local_epochs" \
    --local-batch-size 10 --clients-per-round 25 \
    --local-dp --l2-norm-clip 10.0 --noise-multiplier 0.5
done
