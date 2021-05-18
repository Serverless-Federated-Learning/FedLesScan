#!/usr/bin/env bash

set -e

# Generate random token
token=$(openssl rand -base64 30)

action_name="client-indep"

wsk -i action update \
  "$action_name" \
  main.py \
  --docker andreasgrafberger/fedless-openwhisk \
  --memory 2048 \
  --timeout 60000 \
  --web raw \
  --web-secure "$token"
  #--concurrency 1

# Print info if deployed successfully
wsk -i action get "$action_name"

# Print url to invoke function
API_HOST=$(wsk -i  property get --apihost -o raw)
NAMESPACE=$(wsk -i  property get --namespace -o raw)
ENDPOINT="https://$API_HOST/api/v1/web/$NAMESPACE/default/$action_name.json"
echo "To invoke function, run:"
echo "curl -X POST -H \"X-Require-Whisk-Auth: $token\" -k $ENDPOINT"
#echo "curl -X POST -k $ENDPOINT"
