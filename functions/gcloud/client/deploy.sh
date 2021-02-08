#!/usr/bin/env bash

set -e

if [[ -z "${GITHUB_AUTH_TOKEN}" ]]; then
  echo "Environment variable GITHUB_AUTH_TOKEN not set. Create a token with read access to repo on GitHub."
  exit 1
fi

COMMIT_HASH=$(git rev-parse HEAD)

# shellcheck disable=SC2140
gcloud functions deploy http \
  --runtime python38 \
  --trigger-http \
  --allow-unauthenticated \
  --memory=2048MB \
  --timeout=300s \
  --max-instances 10 \
  --set-build-env-vars GIT_COMMIT_IDENTIFIER="@$COMMIT_HASH",GITHUB_AUTH_TOKEN="$GITHUB_AUTH_TOKEN"

