#!/usr/bin/env bash

# More Clients per round
python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs 100 --local-epochs 10 --local-batch-size 10 --clients-per-round 25 --no-local-dp

# Local DP Clip 2.0
python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs 100 --local-epochs 10 --local-batch-size 10 --clients-per-round 25 \
  --local-dp --l2-norm-clip 2.0 --noise-multiplier 1.0

# Local DP Clip 1.0
python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs 100 --local-epochs 10 --local-batch-size 10 --clients-per-round 25 \
  --local-dp --l2-norm-clip 1.0 --noise-multiplier 1.0

# Local DP Clip 4.0
python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs 100 --local-epochs 10 --local-batch-size 10 --clients-per-round 25 \
  --local-dp --l2-norm-clip 2.0 --noise-multiplier 1.0

# Local DP Clip 2.0 Noise Multiplier 2.0
python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs 100 --local-epochs 10 --local-batch-size 10 --clients-per-round 25 \
  --local-dp --l2-norm-clip 2.0 --noise-multiplier 2.0
