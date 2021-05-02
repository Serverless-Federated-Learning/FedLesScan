#!/usr/bin/env bash

mprof run fedavg-memory.py --stream --small-models
mprof plot --output "mem-usage-stream-small.pdf"

mprof run fedavg-memory.py --no-stream --small-models
mprof plot --output "mem-usage-no-stream-small.pdf"

mprof run fedavg-memory.py --stream --large-models
mprof plot --output "mem-usage-stream-large.pdf"

mprof run fedavg-memory.py --no-stream --large-models
mprof plot --output "mem-usage-no-stream-large.pdf"