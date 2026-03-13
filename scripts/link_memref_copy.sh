#!/bin/bash
# Link the memrefCopy function definition into an LLVM IR file.
#
# Usage: link_memref_copy.sh <target.ll>
#
# Compiles scripts/lib/memref_copy.c to LLVM IR, strips target metadata
# to match MLIR-generated IR, and links it into the given .ll file in-place.

set -e -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <target.ll>"
  exit 1
fi

TARGET_LL="$1"

if [ ! -f "$TARGET_LL" ]; then
  echo "Error: $TARGET_LL not found"
  exit 1
fi

# Check if memrefCopy is even referenced in the target file
if ! grep -q '@memrefCopy' "$TARGET_LL"; then
  echo "Skipping: no memrefCopy reference found in $TARGET_LL"
  exit 0
fi

# Check if memrefCopy is already defined (not just declared)
if grep -q 'define.*@memrefCopy' "$TARGET_LL"; then
  echo "Skipping: memrefCopy already defined in $TARGET_LL"
  exit 0
fi

SRC="${SCRIPT_DIR}/lib/memref_copy.c"

# Use a temp directory for intermediate files
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

UNOPT_LL="${TMPDIR}/memref_copy_unopt.ll"
CLEAN_LL="${TMPDIR}/memref_copy.ll"
COMBINED_LL="${TMPDIR}/combined.ll"

# Step 1: clang -> unoptimized LLVM IR
$DOCKER_RUN clang -S -emit-llvm -O0 -o "${UNOPT_LL}" "${SRC}"

# Step 2: opt pass (keep at O0 to preserve structure for HLS)
$DOCKER_RUN opt -S -O0 -o "${CLEAN_LL}" "${UNOPT_LL}"

# Step 3: Strip target datalayout and target triple to match MLIR-generated IR
sed -i '/^target datalayout/d; /^target triple/d' "${CLEAN_LL}"

# Step 4: Link memref_copy definition into the target
$DOCKER_RUN llvm-link -S "${CLEAN_LL}" "${TARGET_LL}" -o "${COMBINED_LL}"

# Step 5: Replace the original file
mv "${COMBINED_LL}" "${TARGET_LL}"
