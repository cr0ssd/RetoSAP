#!/usr/bin/env bash
set -euo pipefail

# Empaqueta un contexto minimo del Worker (solo src/ + dependencias de inferencia)
# para evitar subir notebooks, dashboard o dependencias de Jupyter a Cloud Foundry.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cf_worker_stage.XXXXXX")"
trap 'rm -rf "$STAGE_DIR"' EXIT

mkdir -p "$STAGE_DIR/src"
rsync -a \
  --exclude='dashboard/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.ipynb' \
  "$REPO_ROOT/src/" "$STAGE_DIR/src/"

cp "$SCRIPT_DIR/requirements.txt" "$STAGE_DIR/requirements.txt"
cp "$SCRIPT_DIR/runtime.txt" "$STAGE_DIR/runtime.txt"
cp "$SCRIPT_DIR/Procfile" "$STAGE_DIR/Procfile"

echo "[INFO] Subiendo paquete ligero del worker a SAP BTP Cloud Foundry..."
cf push -f "$SCRIPT_DIR/manifest.yml" -p "$STAGE_DIR" --no-start "$@"
