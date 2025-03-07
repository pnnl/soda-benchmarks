#!/bin/bash

set -e

echo "Testing tflite flow\n"
pwd
cd models/tflite

for d in */ ; do
    echo "Testing--$d"
    cd $d
    make -s
    make clean -s
    cd ..
    echo "$d: success\n"
done

