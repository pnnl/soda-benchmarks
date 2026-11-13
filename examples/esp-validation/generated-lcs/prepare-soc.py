#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Install generated LCS RTL in an isolated copy of the validated VC707 SoC."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("generated", type=Path)
p.add_argument("esp_source", type=Path)
p.add_argument("compiled_soc", type=Path)
p.add_argument("destination", type=Path)
p.add_argument("--wrapper", choices=["thin", "buffered"], default="thin")
a = p.parse_args()
generated, source, template, root = (
    x.resolve() for x in (a.generated, a.esp_source, a.compiled_soc, a.destination)
)
if root.exists():
    raise SystemExit(f"Use a fresh isolated destination: {root}")
here = Path(__file__).resolve().parent
wrapper = Path(subprocess.check_output([
    "python3", str(here.parents[2]/"scripts/lcs/native_dma_contract.py"),
    "select", str(generated), "--wrapper", a.wrapper], text=True).strip())
root.mkdir(parents=True)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


for name in ("rtl", "soft", "constraints", ".cache"):
    link(source / name, root / name)
# ESP regenerates utils/flist entries even while evaluating Make prerequisites.
# Keep those writes local to this experiment, rather than following a symlink.
shutil.copytree(source / "utils", root / "utils", symlinks=True)
template_root = template.parent.parent
for tool in (source / "tools").iterdir():
    if tool.name in ("socgen", "socketgen"):
        shutil.copytree(template_root / "tools" / tool.name,
                        root / "tools" / tool.name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        link(tool, root / "tools" / tool.name)
for folder in (source / "accelerators").iterdir():
    dest = root / "accelerators" / folder.name
    if folder.name in ("bambu_hls", "third-party"):
        dest.mkdir(parents=True, exist_ok=True)
    else:
        link(folder, dest)
(root / "accelerators/bambu_hls/sqr_bambu").mkdir()
for item in (source / "tech/virtex7").iterdir():
    if item.name != "acc":
        link(item, root / "tech/virtex7" / item.name)
acc = root / "tech/virtex7/acc/sqr_bambu"
for name in ("forward_kernel.v", "panda_libtech.v"):
    copy(generated / "bambu_syn" / name, acc / name)
for mem in (generated / "bambu_syn").glob("*.mem"):
    copy(mem, acc / mem.name)
copy(template_root / "tech/virtex7/acc/sqr_bambu/sqr_bambu.xml",
     acc / "sqr_bambu.xml")
copy(wrapper,
     acc / "sqr_bambu_basic_dma64/sqr_bambu_basic_dma64.sv")
soc = root / "socs/vc707"
soc.mkdir(parents=True)
link(source / "socs/defconfig", root / "socs/defconfig")
for name in ("Makefile", "top.vhd", "testbench.vhd", "vsim.tcl"):
    copy(template / name, soc / name)
for folder in ("socgen/esp", "socgen/grlib", "socketgen"):
    shutil.copytree(template / folder, soc / folder,
                    ignore=shutil.ignore_patterns("__pycache__"))
# This is the exact generated socket/configuration used by the simulation
# fixture. Only the separately compiled accelerator library changes.
for folder in ("socgen/esp", "socgen/grlib", "socketgen"):
    for original in (template / folder).glob("*.vhd"):
        assert original.read_bytes() == (soc / folder / original.name).read_bytes()
manifest = {
    "generated": str(generated), "esp_source": str(source),
    "simulation_soc_template": str(template), "soc": str(soc),
    "device_id": "0x07a", "compatible": "sld,sqr_bambu",
    "wrapper": a.wrapper,
    "rtl_sha256": {
        str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in acc.rglob("*") if file.is_file()
    },
}
(root / "integration.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(soc)
