#!/usr/bin/env bash

set -e

if [[ -z "${GITHUB_AUTH_TOKEN}" ]]; then
  echo "Environment variable GITHUB_AUTH_TOKEN not set. Create a token with read access to repo on GitHub."
  exit 1
fi

COMMIT_HASH=$(git rev-parse HEAD)

for i in {1..10}; do
  function_name="http-indep-${i}"
  echo "Deploying function $function_name"
  # shellcheck disable=SC2140
  gcloud functions deploy "$function_name" \
    --runtime python38 \
    --trigger-http \
    --entry-point="http" \
    --allow-unauthenticated \
    --memory=2048MB \
    --timeout=300s \
    --max-instances 50 \
    --set-build-env-vars GIT_COMMIT_IDENTIFIER="@$COMMIT_HASH",GITHUB_AUTH_TOKEN="$GITHUB_AUTH_TOKEN" &
  if [ "$i" -gt 20 ]
  then
    sleep 5
  fi
done
