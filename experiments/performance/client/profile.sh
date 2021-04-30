#!/usr/bin/env bash
set -e

pip install snakeviz
python profile_client_handler.py
snakeviz program.prof
