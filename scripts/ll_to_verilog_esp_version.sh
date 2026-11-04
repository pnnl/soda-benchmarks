#!/bin/bash -e

set -e -o pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input.ll> <top-fname>" >&2
  exit 1
fi

INPUT=$(realpath "$1")
DIRNAME_INPUT="$(dirname "${INPUT}")"
TOP_FNAME="$2"

mkdir -p "$DIRNAME_INPUT"/bambu_syn
cd "$DIRNAME_INPUT"/bambu_syn

bambu "$INPUT" \
  --architecture-xml=../architecture.xml \
  --top-fname="$TOP_FNAME" \
  --compiler=I386_CLANG19 \
  --generate-interface=INFER
