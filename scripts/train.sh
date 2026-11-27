#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/train.sh [--config FILE] [--dry-run] [key=value ...]

Start the RECAST training driver in the foreground. Paths are relative
to the repository root unless specified as absolute paths.

  --config FILE  Select a training YAML (default: configs/train_video_trsi_fast.yaml).
  --dry-run      Print the command without importing or launching training.
  -h, --help     Show this help.

Override config values with key=value arguments. Set PYTHON_BIN to select
another Python executable. For a cluster started separately, set RAY_ADDRESS.
EOF
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
config="configs/train_video_trsi_fast.yaml"
dry_run=false
overrides=()

while (( $# )); do
  case "$1" in
    --config)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        printf 'Missing file for --config.\n' >&2
        exit 2
      fi
      config="$2"
      shift
      ;;
    --dry-run)
      dry_run=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    config=*)
      printf 'Select the config file with --config FILE.\n' >&2
      exit 2
      ;;
    -*)
      printf 'Unknown option: %s\n' "$1" >&2
      exit 2
      ;;
    ?*=*)
      overrides+=("$1")
      ;;
    *)
      printf 'Expected a key=value override, got: %s\n' "$1" >&2
      exit 2
      ;;
  esac
  shift
done

cd -- "$repo_root"
if [[ ! -f "$config" || ! -r "$config" ]]; then
  printf 'Cannot read training config: %s\n' "$config" >&2
  exit 2
fi

command=("${PYTHON_BIN:-python3}" -m verl.trainer.trsi_main "config=$config")
if (( ${#overrides[@]} )); then
  command+=("${overrides[@]}")
fi

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
if "$dry_run"; then
  printf 'cd %q && NCCL_TIMEOUT=%q' "$repo_root" "$NCCL_TIMEOUT"
  printf ' %q' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
