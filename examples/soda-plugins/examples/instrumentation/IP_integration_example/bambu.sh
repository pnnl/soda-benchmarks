#!/bin/bash
script=$(readlink -e $0)
root_dir=$(dirname $script)

rm -rf IP_integration_hls
mkdir -p IP_integration_hls
cd IP_integration_hls
echo "#synthesis with VIVADO RTL and simulation"
timeout 2h bambu $root_dir/top.c --top-fname=my_ip $root_dir/module_lib.xml $root_dir/constraints_STD.xml \
   --experimental-setup=BAMBU --memory-allocation-policy=ALL_BRAM -O2 \
   -v3 \
   --C-no-parse=$root_dir/module1.c,$root_dir/module2.c,$root_dir/printer1.c,$root_dir/printer2.c,$root_dir/sodaInstrAssertLessThen.c \
   --file-input-data=$root_dir/module1.v,$root_dir/module2.v,$root_dir/printer1.v,$root_dir/printer2.v,$root_dir/sodaInstrAssertLessThen.v \
   --generate-tb=$root_dir/main_test.c --simulate --simulator=VERILATOR --print-dot "$@" --generate-components-library
return_value=$?
if test $return_value != 0; then
   exit $return_value
fi
cd ..
exit 0
