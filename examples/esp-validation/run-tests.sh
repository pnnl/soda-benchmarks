#!/usr/bin/env bash
# Convenience entry point; the lit RUN lines are the only test definitions.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${SODAP_BUILD_DIR:-$repo/examples/soda-plugins/build}
lit=${LLVM_LIT:-}
if [[ -z "$lit" ]]; then
  lit=$(command -v llvm-lit || command -v lit || true)
fi
if [[ -z "$lit" ]] || ! command -v "$lit" >/dev/null 2>&1; then
  printf 'Set LLVM_LIT to an installed llvm-lit/lit executable. See docs/ESPValidation.md.\n' >&2
  exit 2
fi
if [[ ! -f "$build/test/lit.site.cfg.py" ]]; then
  printf 'Configure/build soda-plugins first (SODAP_BUILD_DIR=%s).\n' "$build" >&2
  exit 2
fi
exec "$lit" -v --filter='sodap/ESP/' "$build/test" "$@"
