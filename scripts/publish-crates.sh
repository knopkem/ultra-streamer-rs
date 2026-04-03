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
  - Already-published crate versions are skipped automatically.
EOF
}

# Returns 0 if the given crate@version already exists on crates.io.
is_published() {
  local crate="$1" version="$2"
  local http_code
  http_code=$(curl -sf -o /dev/null -w "%{http_code}" \
    -A "publish-crates.sh/1.0" \
    "https://crates.io/api/v1/crates/${crate}/${version}" 2>/dev/null || true)
  [[ "$http_code" == "200" ]]
}

# Read the version for a given crate from the workspace Cargo.toml / package manifest.
crate_version() {
  local crate="$1"
  cargo metadata --no-deps --format-version 1 2>/dev/null \
    | grep -A3 "\"name\":\"${crate}\"" \
    | grep '"version"' \
    | head -1 \
    | sed 's/.*"version":"\([^"]*\)".*/\1/'
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
  version="$(crate_version "$crate")"
  if [[ "$mode" == "--publish" ]] && is_published "$crate" "$version"; then
    echo "--- skipping $crate@$version (already on crates.io) ---"
    continue
  fi
  echo "--- cargo publish -p $crate ${publish_args[*]+"${publish_args[*]}"} ---"
  cargo publish -p "$crate" ${publish_args[@]+"${publish_args[@]}"}
done

