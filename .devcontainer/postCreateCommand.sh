#!/bin/bash
set -e

echo "Installing or updating dev tools..."
pushd /tmp
# pixi is provided by the soda image (/usr/local/bin/pixi)
# curl -fsSL https://claude.ai/install.sh | bash
popd
echo "Completed."