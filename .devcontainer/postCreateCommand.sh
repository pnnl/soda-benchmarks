#!/bin/bash
set -e

echo "Installing or updating dev tools..."
pushd /tmp
curl -fsSL https://pixi.sh/install.sh | sh
# curl -fsSL https://claude.ai/install.sh | bash
popd
echo "Completed."