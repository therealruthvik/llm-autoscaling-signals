#!/usr/bin/env bash
# Create the kind cluster. Safe to rerun: skips creation if it already exists.
set -euo pipefail
cd "$(dirname "$0")"

if kind get clusters | grep -qx autoscale; then
  echo "cluster 'autoscale' already exists"
else
  kind create cluster --config kind-config.yaml
fi
kubectl cluster-info --context kind-autoscale
# Helm installs fail with RBAC 'forbidden' errors if run before the node is Ready.
kubectl wait --for=condition=Ready node --all --timeout=180s
kubectl get nodes
