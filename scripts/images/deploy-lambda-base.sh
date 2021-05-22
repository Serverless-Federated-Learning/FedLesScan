#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Switching to project root directory $ROOT_DIR"
cd "$ROOT_DIR"

# Build python project
echo "Build package"
python setup.py bdist_wheel

# Building image
echo "Build and Push Image"
docker build -f ./images/lambda/base/Dockerfile -t fedless-lambda-git  .
docker tag fedless-lambda-git andreasgrafberger/fedless:lambda
docker push andreasgrafberger/fedless:lambda
docker build -f ./images/lambda/base/AVX512.Dockerfile -t fedless-lambda-avx512-git  .
docker tag fedless-lambda-avx512-git andreasgrafberger/fedless:lambda-avx512
docker push andreasgrafberger/fedless:lambda-avx512
