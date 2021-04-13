#!/usr/bin/env bash

set -e

# Generate random token
token=$(openssl rand -base64 30)

wsk -i action update \
  client \
  main.py \
  --docker andreasgrafberger/fedless-openwhisk \
  --memory 2048 \
  --timeout 60000 \
  --web raw \
  --web-secure "$token" \
  --concurrency 1

# Print info if deployed successfully
wsk -i action get client

# Print url to invoke function
API_HOST=$(wsk -i  property get --apihost -o raw)
NAMESPACE=$(wsk -i  property get --namespace -o raw)
ENDPOINT="https://$API_HOST/api/v1/web/$NAMESPACE/default/client.json"
echo "To invoke function, run:"
echo "curl -X POST -H \"X-Require-Whisk-Auth: $token\" -k $ENDPOINT"
