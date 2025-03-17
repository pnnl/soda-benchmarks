#!/bin/bash

set -e

echo "Testing pytorch flow\n"
pwd
cd ../models/pytorch

echo "Testing--mobilenetv3_small"
cd mobilenetv3_small
make -s
make clean -s
cd ..
echo "mobilenetv3_small: success\n"

