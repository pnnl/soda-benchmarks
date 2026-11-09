#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Match the compiled invocation plan to CPU and physical accelerator logs."""
import json
from pathlib import Path
import re
import sys
plan = json.loads(Path(sys.argv[1]).read_text())
logfile = Path(sys.argv[2])
log = logfile.read_text()
assert 'TEST PASSED' in log and 'TEST FAILED' not in log
assert '** Fatal' not in log and '** Error' not in log
assert 'Failed to open readmem' not in log
assert 'NN_SYSTEM_SIM_TIME_LIMIT' not in log
assert '** Failure: Program Completed!' in log
assert all('Program Completed!' in line for line in log.splitlines() if '** Failure:' in line)
expected = [x['operator'] for x in plan['invocations'] if x['execution']=='accelerator']
actual = re.findall(r'REUSABLE_OP launch=(\d+) kind=(\w+) instance=(\d+) socket=(\d+)', log)
assert actual == [(str(i), kind, '0', '0') for i, kind in enumerate(expected)], (expected, actual)
count = len(expected)
assert count and len(re.findall('REUSABLE_OP open socket=', log)) == 1
assert re.findall(r'REUSABLE_OP close socket=(\d+) invocations=(\d+)', log) == [('0', str(count))]
assert re.findall(r'REUSABLE_OP complete=(\d+) socket=(\d+)', log) == [(str(i), '0') for i in range(1,count+1)]
for marker in ('START','RESULT'):
    assert re.findall(rf'GEMM_BOUNDARY {marker} invocation=(\d+)',log) == [str(i) for i in range(1,count+1)]
result = re.findall(r'NN_RESULT outputs=(\d+) errors=(\d+) max_error_e9=(\d+) top1=(\d+) reference_top1=(\d+) cycles=(\d+)', log)
assert len(result)==1 and result[0][1]=='0', result
report={'status':'PASS','execution':'full-system RTL','operators':expected,
        'hardware_invocations':count,'outputs':int(result[0][0]),'max_error_e9':int(result[0][2]),
        'cpu_cycles':int(result[0][5])}
logfile.with_name('validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
