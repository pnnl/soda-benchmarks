#!/usr/bin/env python3
"""Bounded analytical proposal search, scored on actual scheduled MLIR.

No HLS or measured timings in the objective. The original divisor-search model
generates proposals at several storage caps; the final schedule accounts for
reuse, recomputation and physical ping-pong copies. This is a heuristic search,
not an exhaustive/global optimum. Both overlap modes use the same selection.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

source, output, pipeline = map(Path,sys.argv[1:])
total=int(os.environ.get('LCS_PLM_BYTES','327680'))
reuse=int(os.environ.get('LCS_REUSE_BUFFER_BUDGET','262144'))
if total<=0 or reuse<0:
    raise SystemExit('Expected positive LCS_PLM_BYTES and nonnegative reuse budget')
proposals=[('plm',n) for n in sorted({max(64,total//d) for d in (32,16,8,4,1)})]
proposals.append(('max_tile',10)) # Include the historical generic schedule.
records=[]
best=None
for i,(kind,value) in enumerate(proposals):
    folder=output/'.tile-search'/str(i)
    folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,_LCS_INTERNAL_CANDIDATE='true',LCS_OVERLAP='true',LCS_SEQUENTIAL='false')
    env.pop('LCS_TILE_SIZES',None)
    env.pop('_LCS_PROPOSAL_PLM_BYTES',None)
    env.pop('_LCS_PROPOSAL_MAX_TILE',None)
    env['_LCS_PROPOSAL_PLM_BYTES' if kind=='plm' else '_LCS_PROPOSAL_MAX_TILE']=str(value)
    record={'proposal':{kind:value},'directory':str(folder)}
    try:
        with (folder/'search.log').open('w') as log:
            run=subprocess.run(['bash',str(pipeline),str(source),str(folder)],env=env,
                               stdout=log,stderr=subprocess.STDOUT,timeout=120)
        record['returncode']=run.returncode
        if run.returncode==0:
            cost=json.loads((folder/'schedule_cost.json').read_text())
            tiles=json.loads((folder/'tile_sizes.txt').read_text())
            score=(cost['dma_cycles'],cost['compute_iterations'],cost['plm_bytes'])
            record.update(status='VALID',cost=cost,tiles=tiles)
            if best is None or score<best[0]: best=(score,tiles,i)
        else:
            record.update(status='REJECTED',diagnostic=(folder/'search.log').read_text()[-1800:])
    except subprocess.TimeoutExpired:
        record.update(status='TIMEOUT')
    records.append(record)
    (output/'tile_search.json').write_text(json.dumps({'plm_budget':total,'reuse_budget':reuse,
         'scope':'Heuristic analytical ranking of scheduled DMA work, not measured latency',
         'selected':best[2] if best else None,'candidates':records},indent=2)+'\n')
if best is None:
    raise SystemExit('No supported scheduled candidate fits the total PLM budget; see tile_search.json')
(output/'selected_tile_sizes.txt').write_text(json.dumps(best[1])+'\n')
print(f'Selected tiles {best[1]}: scheduled DMA cost {best[0][0]}, storage {best[0][2]} bytes',flush=True)
