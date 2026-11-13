#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Program a specified VC707 target and validate one compiled network via UART."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("soc",type=Path)
parser.add_argument("application",type=Path)
parser.add_argument("bitstream",type=Path)
parser.add_argument("output",type=Path)
parser.add_argument("--uart-host",default="william@espdev")
parser.add_argument("--uart-log",required=True)
parser.add_argument("--jtag-host",default="espdev")
parser.add_argument("--jtag-port",type=int,default=3131)
parser.add_argument("--esplink-ip",default="128.59.22.75")
parser.add_argument("--esplink-port",type=int,default=46313)
parser.add_argument("--timeout",type=float,default=1800)
parser.add_argument("--resume",action="store_true",
                    help="Resume UART monitoring of an existing RUNNING result without resetting the board")
args=parser.parse_args()
soc,app,bit,out=(p.resolve() for p in (args.soc,args.application,args.bitstream,args.output))
assert bit.is_file() and (app/"nn_app.exe").is_file()
timing=dict(line.split("\t") for line in (bit.parent/"reports/timing.tsv").read_text().splitlines()[1:])
assert all(float(timing[key])>=0 for key in ("STATS.WNS","STATS.WHS")), "FPGA timing has not closed"
memory=json.loads((app/"memory.json").read_text())
assert memory["dma_base"]=="0xa0200000"
execution=memory.get("execution","FPGA")
assert execution in ("FPGA","CPU_ONLY")
prefix="CPU_OP" if execution=="CPU_ONLY" else "REUSABLE_OP"
assert args.timeout>0
plan=json.loads((app/"plan.json").read_text())
expected=[x["operator"] for x in plan["invocations"] if x["execution"]=="accelerator"]
outputs=1
for d in plan["output_shape"]:
    outputs*=d
if not args.resume:
    out.mkdir(parents=True,exist_ok=False)
here=Path(__file__).resolve().parent
env=dict(os.environ,PATH="/opt/xilinx/Vivado/2023.2/bin:/opt/riscv/bin:"+os.environ["PATH"],
         XILINX_VIVADO="/opt/xilinx/Vivado/2023.2",PYTHONDONTWRITEBYTECODE="1")
state={"status":"PREPARING","soc":str(soc),"jtag":f"{args.jtag_host}:{args.jtag_port}",
       "uart_host":args.uart_host,"uart_log":args.uart_log,
       "artifacts":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (bit,app/"nn_app.exe",app/"plan.json",app/"memory.json")},
       "execution":execution,"memory":memory}
if args.resume:
    previous=json.loads((out/"validation.json").read_text())
    assert previous["status"]=="RUNNING", "Only an active run can be resumed"
    for key in ("soc","uart_host","uart_log","artifacts","execution","memory"):
        assert previous[key]==state[key], f"Resume mismatch: {key}"
    assert isinstance(previous["uart_offset"],int) and previous["uart_offset"]>=0
    state=previous

def save():
    state["updated"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/"validation.json").write_text(json.dumps(state,indent=2)+"\n")

def remote(command):
    return subprocess.check_output(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8",
                                    args.uart_host,command],timeout=15)

def run(command,leaf,timeout=300):
    with (out/leaf).open("a") as stream:
        stream.write(shlex.join([str(x) for x in command])+"\n")
        stream.flush()
        subprocess.run([str(x) for x in command],cwd=soc,env=env,stdout=stream,
                       stderr=subprocess.STDOUT,timeout=timeout,check=True)

def parse(raw):
    text=re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]","",raw.decode(errors="replace")).replace("\r","")
    banners=list(re.finditer(r"\bN?N_BEGIN outputs=(\d+) arena_bytes=(\d+)",text))
    if banners:
        banner=banners[-1]
        assert int(banner[1])==outputs
        text=text[banner.start():]
    elif execution=="CPU_ONLY" and "NN_MODE execution=CPU_ONLY arena=cacheable" in text:
        # Some boots lose the first UART line. The explicit mode marker,
        # complete operator sequence and final result still identify the run.
        text=text[text.rfind("NN_MODE execution=CPU_ONLY arena=cacheable"):]
    else:
        return None
    if any(marker in text for marker in ("TEST FAILED","NN_TRAP","MISMATCH")):
        raise RuntimeError("Board application failed; inspect uart.log")
    result=re.search(r"NN_RESULT outputs=(\d+) errors=(\d+) max_error_e9=(\d+) top1=(\d+) reference_top1=(\d+) cycles=(\d+)",text)
    if "TEST PASSED" not in text or not result:
        return None
    assert int(result[1])==outputs and result[2]=="0"
    if execution=="CPU_ONLY":
        assert "NN_MODE execution=CPU_ONLY arena=cacheable" in text
        assert "REUSABLE_OP launch=" not in text
    launches=re.findall(prefix+r" launch=(\d+) kind=(\w+) instance=(\d+) socket=(\d+)",text)
    assert launches==[(str(i),kind,"0","0") for i,kind in enumerate(expected)], launches
    assert re.findall(prefix+r" complete=(\d+) socket=(\d+)",text)==[(str(i),"0") for i in range(1,len(expected)+1)]
    assert re.findall(prefix+r" close socket=(\d+) invocations=(\d+)",text)==[("0",str(len(expected)))]
    return {"status":"PASS","execution":execution,
            "hardware_invocations":len(expected) if execution=="FPGA" else 0,
            "software_operator_invocations":len(expected) if execution=="CPU_ONLY" else 0,
            "operators":expected,"outputs":outputs,"errors":0,"max_error_e9":int(result[3]),
            "top1":int(result[4]),"reference_top1":int(result[5]),"cpu_cycles":int(result[6])}

save()
try:
    # Require an existing recording before changing the FPGA or resetting it.
    int(remote("wc -c < "+shlex.quote(args.uart_log)))
    if args.resume:
        offset=state["uart_offset"]
    else:
        run(["vivado","-mode","batch","-nojournal","-log",out/"vivado.log","-source",
             here/"../generated-lcs/program-board.tcl","-tclargs",args.jtag_host,args.jtag_port,bit],"program.log")
        assert "LCS_FPGA_PROGRAMMED" in (out/"program.log").read_text()
        state["status"]="LOADING"
        save()
        time.sleep(5)
        run(["make","-C",soc,"soft","esplink",f"TEST_PROGRAM={app/'nn_app.exe'}",
             f"ESPLINK_IP={args.esplink_ip}",f"ESPLINK_PORT={args.esplink_port}"],"load.log")
        esplink=soc/"socgen/esp/esplink"
        run([esplink,"--reset"],"load.log",30)
        run([esplink,"--brom","-i",soc/"soft-build/ariane/prom.bin"],"load.log",60)
        run([esplink,"--dram","-i",soc/"soft-build/ariane/systest.bin"],"load.log",600)
        time.sleep(1)
        offset=int(remote("wc -c < "+shlex.quote(args.uart_log)))
        state.update(status="RUNNING",uart_offset=offset)
        save()
        run([esplink,"--reset"],"load.log",30)
    deadline=time.monotonic()+args.timeout
    while True:
        raw=remote(f"tail -c +{offset+1} -- "+shlex.quote(args.uart_log))
        (out/"uart.log").write_bytes(raw)
        result=parse(raw)
        if result:
            state.update(result)
            save()
            print(json.dumps(result,indent=2))
            break
        if time.monotonic()>deadline:
            raise TimeoutError("No complete network validation before UART deadline")
        time.sleep(2)
except Exception as error:
    state.update(status="FAIL",error=str(error))
    save()
    raise
