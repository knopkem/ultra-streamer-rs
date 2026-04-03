#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/publish-crates.sh list
  scripts/publish-crates.sh <stage1|stage2|stage3> [--dry-run|--publish] [--allow-dirty]

Stages:
  stage1  ustreamer-proto ustreamer-capture
  stage2  ustreamer-input ustreamer-quality ustreamer-transport ustreamer-encode
  stage3  ustreamer-app

Notes:
  - Run stages in order.
  - Wait until the previous stage is visible on crates.io before moving on.
  - `--dry-run` is the default mode.
  - `--allow-dirty` is useful for local rehearsal only; do not use it for real releases.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

stage="$1"
shift

mode="--dry-run"
allow_dirty=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--publish)
      mode="$1"
      ;;
    --allow-dirty)
      allow_dirty=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unrecognized argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

case "$stage" in
  list)
    cat <<'EOF'
Release order:
  stage1: ustreamer-proto ustreamer-capture
  stage2: ustreamer-input ustreamer-quality ustreamer-transport ustreamer-encode
  stage3: ustreamer-app

Wait for each stage to be visible on crates.io before starting the next one.
EOF
    exit 0
    ;;
  stage1)
    crates=(ustreamer-proto ustreamer-capture)
    ;;
  stage2)
    crates=(ustreamer-input ustreamer-quality ustreamer-transport ustreamer-encode)
    ;;
  stage3)
    crates=(ustreamer-app)
    ;;
  *)
    echo "error: unknown stage '$stage'" >&2
    usage
    exit 1
    ;;
esac

publish_args=()
if [[ "$mode" == "--dry-run" ]]; then
  publish_args+=(--dry-run)
fi
if [[ $allow_dirty -eq 1 ]]; then
  publish_args+=(--allow-dirty)
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "Running $stage in ${mode#--} mode"
echo "Crates: ${crates[*]}"

for crate in "${crates[@]}"; do
  echo "--- cargo publish -p $crate ${publish_args[*]+"${publish_args[*]}"} ---"
  cargo publish -p "$crate" ${publish_args[@]+"${publish_args[@]}"}
done
