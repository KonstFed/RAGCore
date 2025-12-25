#!/usr/bin/env bash
set -euo pipefail

docker pull qdrant/qdrant
docker build -t ragcore-app -f docker/Dockerfile .
