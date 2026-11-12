#!/usr/bin/env python3
"""Fresh candidate hardware and software in the validated native 2x2 ESP fixture."""
import os,argparse,hashlib,json,re,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('kernel');p.add_argument('--run',required=True)
p.add_argument('--variant',choices=['overlap','nonoverlap'],default='overlap')
p.add_argument('--wrapper',choices=['thin','buffered'],default='thin')
a=p.parse_args();tool=Path(__file__).resolve().parent;repo=tool.parents[2]
root=Path(os.environ.get('ESP_REVIEW_ARTIFACTS',repo/'build/final-review')).resolve()
platform=root/'platform/esp/socs/vc707'
vendor=root/'platform/vendor'
libs=vendor
app=root/'apps/MINI'/a.kernel;gen=root/'generated/MINI'/a.kernel/a.variant
wrapper=(repo/'examples/esp-validation/native-dma/thin_dma64.sv' if a.wrapper=='thin' else repo/'examples/esp-validation/generated-gemm/sqr_bambu_basic_dma64.sv')
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
spec=json.loads((gen.parent/'fixture.json').read_text())
component_path=gen.parent/'components'/a.variant/a.wrapper/'result.json'
component=json.loads(component_path.read_text());assert component['status']=='PASS'
sources=[gen/'bambu_syn/forward_kernel.v',gen/'bambu_syn/panda_libtech.v',wrapper]
for q in sources:
 matches=[v for k,v in component['sources'].items() if Path(k).name==q.name]
 assert matches==[sha(q)],f'Stale component evidence: {q}'
build=json.loads((app/'build.json').read_text())
assert sha(app/'app.exe')==build['elf_sha256']
assert sha(gen.parent/'overlap/06_kernel_riscv.o')==build['host_object_sha256']
assert (gen/'05_kernel_riscv.ll').read_bytes()==(gen.parent/'overlap/05_kernel_riscv.ll').read_bytes()
out=root/'systems'/a.run;out.mkdir(parents=True,exist_ok=False)
sim=out/'modelsim';sim.mkdir()
record={'status':'RUNNING','kernel':a.kernel,'variant':a.variant,'wrapper':a.wrapper,
        'scope':'Fresh MINI candidate IP and host, full-system native 2x2 ESP RTL; not FPGA',
        'sources':{str(q):sha(q) for q in sources},'elf_sha256':sha(app/'app.exe'),
        'component_result_sha256':sha(component_path),'platform_sha256':sha(platform/'modelsim/compiled.json'),
        'vendor_sha256':sha(vendor/'compiled.json')}
def save(): (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
save();start=time.monotonic()
def run(args,log,cwd=sim):
 with (sim/log).open('w') as f:subprocess.run(list(map(str,args)),cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
try:
 shutil.copytree(platform/'modelsim/work',sim/'work')
 shutil.copy2(root/'platform/prom.srec',sim/'prom.srec')
 for name in ('ram.srec','app.exe'):shutil.copy2(app/name,sim/name)
 for q in (gen/'bambu_syn').glob('*.mem'):shutil.copy2(q,sim/q.name)
 (sim/'modelsim.ini').write_text('[Library]\nothers = '+'/opt/cad/modelsim/modelsim.ini'+'\nwork = work\nsqr_bambu = accelerator\nunisim = '+str(vendor/'unisim')+'\nunisims_ver = '+str(libs/'unisims_ver')+'\nsecureip_ver = '+str(libs/'secureip_ver')+'\n')
 run(['vlib','accelerator'],'vlib.log')
 run(['vlog','-sv','+define+SODAP_LCS_GENERIC','-work','sqr_bambu',*sources],'compile.log')
 ram=out/'soft-build/ariane';ram.mkdir(parents=True)
 shutil.copy2(app/'app.bin',ram/'systest.bin')
 run(['python3',platform.parent.parent/'utils/scripts/file_handling/bin2txt_vhx.py','64','ariane'],'ram.log',out)
 for name in ('trace_hart_0.log','trace_hart_0_commit.log'):(sim/name).symlink_to('/dev/null')
 (sim/'run.tcl').write_text('onerror {quit -code 2 -f}\nonbreak {quit -code 0 -f}\nrun 200 ms\nputs "PB_SYSTEM_TIME_LIMIT"\nquit -code 3 -f\n')
 command=['vsim','-c','-suppress','3812,2697,8617,151,143,8386,3009,2685,2718',
  '-L','secureip_ver','-L','unisims_ver','-L','work','-L','sqr_bambu','-uvmcontrol=disable','-t','fs','+notimingchecks',
  'work.testbench','work.glbl','-G/testbench/soc_inst/gen_mig_model/mig_axiram/kbytes=16384',
  '-do','do run.tcl','-l','simulation.log','+gemm_timeout=5000000']
 record['command']=command;run(command,'console.log')
 log=(sim/'simulation.log').read_text(errors='replace')
 assert '** Failure: Program Completed!' in log
 assert all('Program Completed!' in s for s in log.splitlines() if '** Failure:' in s)
 assert not any(s in log for s in ('** Error','** Fatal','PB_SYSTEM_TIME_LIMIT','TEST FAILED','PB_ERROR','PB_TRAP','is not bound'))
 assert f'PB_BEGIN kernel={a.kernel} dataset=MINI dtype=float32 mode=esp' in log and 'TEST PASSED' in log
 results=re.findall(r'PB_RESULT outputs=(\d+) errors=(\d+) guard_check=0 .* runtime_returned=1',log)
 assert len(results)==1 and tuple(map(int,results[0]))==(spec['outputs'],0)
 cycles=re.findall(r'BOUNDARY RESULT .*?cycles=(\d+)',log);assert len(cycles)==1
 assert all(sha(Path(q))==v for q,v in record['sources'].items())
 assert sha(app/'app.exe')==record['elf_sha256']
 record.update(status='PASS',outputs=spec['outputs'],accelerator_cycles=int(cycles[0]),log_sha256=sha(sim/'simulation.log'))
except Exception as exc:
 record.update(status='FAIL',error=repr(exc));raise
finally:
 record['seconds']=time.monotonic()-start;save();print(a.kernel,record['status'],record.get('accelerator_cycles'),flush=True)
