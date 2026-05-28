#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
checkpoint_dir="${CHECKPOINT_DIR:-${repo_root}/checkpoints}"
depth_anything_url="${DEPTH_ANYTHING_URL:-https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/download_dav2_checkpoint.sh

Downloads to checkpoints/:
  depth_anything_v2_metric_hypersim_vitl.pth

This checkpoint is only required for training from the pretrained DAV2 backbone.

Environment overrides:
  CHECKPOINT_DIR=<path>
  DEPTH_ANYTHING_URL=<url>
USAGE
}

if [[ $# -ne 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  if [[ $# -eq 1 && ( "${1:-}" == "-h" || "${1:-}" == "--help" ) ]]; then
    exit 0
  fi
  exit 1
fi

mkdir -p "$checkpoint_dir"

dest="${checkpoint_dir}/depth_anything_v2_metric_hypersim_vitl.pth"
tmp="${dest}.partial"

if [[ -s "$dest" ]]; then
  echo "Already exists: $dest"
  exit 0
fi

echo "Downloading: $dest"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$tmp" "$depth_anything_url"
elif command -v wget >/dev/null 2>&1; then
  wget -c -O "$tmp" "$depth_anything_url"
else
  echo "Error: install curl or wget to download checkpoints." >&2
  exit 1
fi

mv "$tmp" "$dest"
echo "DAV2 checkpoint ready: $dest"
