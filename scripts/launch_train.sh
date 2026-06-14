#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/main.yaml}"
python -m pedolgov.gateway stratify "$CONFIG"
python -m pedolgov.gateway calibrate "$CONFIG"
