#!/usr/bin/env bash
set -e

pip install snakeviz

python profile_client_handler.py --out out.prof

snakeviz out.prof


