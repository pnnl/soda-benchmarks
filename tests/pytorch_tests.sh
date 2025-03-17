#!/bin/bash

set -e

echo "Testing pytorch flow\n"
pwd
cd ../models/pytorch

for d in */ ; do
    echo "Testing--$d"
    cd $d
    make -s
    make clean -s
    cd ..
    echo "$d: success\n"
done

