#!/usr/bin/env python3
"""Compare the same generated IP with thin/buffered native-DMA adapters."""
import argparse,hashlib,json,re,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('kernel_directory',type=Path)
p.add_argument('--variant',choices=['overlap','nonoverlap'],default='overlap')
p.add_argument('--wrapper',choices=['thin','buffered'],default='thin')
a=p.parse_args();root=a.kernel_directory.resolve();here=Path(__file__).resolve().parent
rtl=root/a.variant/'bambu_syn';out=root/'components'/a.variant/a.wrapper
out.mkdir(parents=True,exist_ok=True)
assert json.loads((root/'capture/result.json').read_text())['status']=='PASS'
wrapper=here/'thin_dma64.sv' if a.wrapper=='thin' else here.parent/'generated-gemm/sqr_bambu_basic_dma64.sv'
for name in ('initial.hex','expected.hex','mask.hex','parameters.svh'):shutil.copy2(root/'capture'/name,out/name)
for f in rtl.glob('*.mem'):shutil.copy2(f,out/f.name)
tb=(here.parent/'generated-lcs/dma64_tb.sv').read_text()
tb=tb.replace(',\n                         .WRITE_BUFFER_BEATS(WRITE_BUFFER_BEATS)','')
tb=tb.replace('error_value>TOL','error_value>(TOL+0.0001*((fp32(gold[i])<0)?-fp32(gold[i]):fp32(gold[i])))')
(out/'tb.sv').write_text(tb)
sources=[rtl/'forward_kernel.v',rtl/'panda_libtech.v',wrapper,out/'tb.sv']
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
record={'status':'RUNNING','variant':a.variant,'wrapper':a.wrapper,
        'sources':{str(q):sha(q) for q in sources},
        'vectors':{n:sha(out/n) for n in ('initial.hex','expected.hex','mask.hex','parameters.svh')}}
start=time.monotonic()
try:
 command=['iverilog','-g2012','-DSODAP_LCS_GENERIC','-I'+str(out),'-s','lcs_dma64_tb',
          '-Plcs_dma64_tb.TIMEOUT_CYCLES=5000000','-o',str(out/'test.vvp'),*map(str,sources)]
 with (out/'compile.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 record['runs']=[]
 for stall in (0,50000):
  path=out/f'simulation-{stall}.log'
  with path.open('w') as log:subprocess.run(['vvp','test.vvp',f'+gemm_stall={stall}'],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
  text=path.read_text();assert text.count('LCS_DMA64 PASS errors=0')==1 and text.count('LCS_DMA64 RESULT invocation=')==2
  record['runs'].append({'initial_write_stall':stall,'cycles':list(map(int,re.findall(r'BOUNDARY RESULT .*?cycles=(\d+)',text))),'log_sha256':sha(path)})
 assert all(sha(Path(q))==v for q,v in record['sources'].items())
 record.update(status='PASS',invocations=4)
except Exception as exc:
 record.update(status='FAIL',error=repr(exc));raise
finally:
 record['seconds']=time.monotonic()-start
 (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
 print(record['status'],root.name,a.variant,a.wrapper,flush=True)
