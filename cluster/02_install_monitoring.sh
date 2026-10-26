#!/usr/bin/env bash
# Install Prometheus and KEDA via Helm. Safe to rerun (helm upgrade --install).
set -euo pipefail
cd "$(dirname "$0")"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

helm upgrade --install prometheus prometheus-community/prometheus \
  --namespace monitoring --create-namespace \
  -f prometheus-values.yaml --wait --timeout 10m

helm upgrade --install keda kedacore/keda \
  --namespace keda --create-namespace \
  --wait --timeout 10m

kubectl get pods -n monitoring
kubectl get pods -n keda
echo "Prometheus UI: http://localhost:9090"
