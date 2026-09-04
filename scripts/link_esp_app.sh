#!/bin/bash
# Finish the ESP application link, inside an ESP checkout.
#
# Usage: link_esp_app.sh <kernel_riscv.o> <output.riscv>
#
# The staged esp-app/ directory (written by ll_to_riscv.sh) is a normal ESP
# baremetal application, so the link is `make` in that directory with ESP's
# common_bare.mk. That needs the ESP tree, its RISC-V toolchain and its
# baremetal support library, none of which ship with this repository -- point
# ESP_ROOT (or DRIVERS) at a checkout to enable this stage.

set -e -o pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <kernel_riscv.o> <output.riscv>" >&2
  exit 1
fi

OUTPUT="$2"
APP_DIR="$(cd "$(dirname "$1")" && pwd)/esp-app"

if [ ! -d "$APP_DIR" ]; then
  echo "Error: $APP_DIR not found; build the 'object' stage first" >&2
  exit 1
fi

DRIVERS="${DRIVERS:-${ESP_ROOT:+$ESP_ROOT/soft/common/drivers}}"

if [ -z "$DRIVERS" ] || [ ! -f "$DRIVERS/common_bare.mk" ]; then
  cat >&2 <<MSG
Error: the ESP baremetal build environment was not found.

  --stage binary links the application with ESP's own common_bare.mk, which
  supplies the RISC-V toolchain, the linker script and the baremetal support
  library. Neither that nor an ESP checkout is part of this container.

  Set one of these and re-run:
    ESP_ROOT=/path/to/esp        (common_bare.mk is looked for under
                                  \$ESP_ROOT/soft/common/drivers)
    DRIVERS=/path/to/esp/soft/common/drivers

  Everything the link needs is already staged and self-contained in
    $APP_DIR
  so building it inside an ESP checkout by hand is equivalent:
    cp -r $APP_DIR <esp>/soft/\$SOC/baremetal/<name> && make
MSG
  exit 1
fi

APP_NAME=$(sed -n 's/^APPNAME *:= *//p' "$APP_DIR/Makefile")
echo "[link_esp_app] DRIVERS=$DRIVERS APPNAME=$APP_NAME"
make -C "$APP_DIR" DRIVERS="$DRIVERS"
cp "$APP_DIR/$APP_NAME.exe" "$OUTPUT" 2>/dev/null || cp "$APP_DIR/$APP_NAME" "$OUTPUT"
echo "[link_esp_app] wrote: $OUTPUT"
