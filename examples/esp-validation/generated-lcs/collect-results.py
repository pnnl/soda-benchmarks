#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Collect campaign evidence and checksum each matching FPGA/application set."""
import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("campaign", type=Path)
p.add_argument("--require-complete", action="store_true")
a = p.parse_args()
root = a.campaign.resolve()
states = json.loads((root / "status.json").read_text())
assert states, "Campaign contains no kernels"


def simulation(path):
    if not path.exists():
        return {}
    text = path.read_text()
    result = {}
    numeric = re.search(r"max error = (\d+) e-9, tolerance = (\d+) e-9", text)
    if numeric:
        result.update(error_e9=int(numeric[1]), tolerance_e9=int(numeric[2]))
    unexpected_failure = any("Program Completed!" not in line for line in text.splitlines()
                             if "** Failure:" in line)
    result["pass"] = ("TEST PASSED" in text and not unexpected_failure and
                      "Failure: Program Completed!" in text and
                      not re.search(r"TEST FAILED|\*\* Fatal|\*\* Error|LCS_SYSTEM_SIM_TIME_LIMIT", text))
    result["status"] = ("PASS" if result["pass"] else "FAIL" if re.search(
        r"TEST FAILED|\*\* Fatal|\*\* Error|LCS_SYSTEM_SIM_TIME_LIMIT", text) else "PENDING")
    result["regions"] = {m[1]: int(m[2]) for m in re.finditer(
        r"^#\s+(total|pack|accel|unpack|alloc|kernel)\s+(\d+)\s+1\s+", text, re.M)}
    boundary = re.search(r"GEMM_BOUNDARY RESULT.*?cycles=(\d+)", text)
    if boundary:
        result["hardware_cycles"] = int(boundary[1])
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


results, rows = {}, []
for name, state in states.items():
    folder = root / name
    offload = simulation(folder / "system-esp/full-system/simulation.log")
    cpu = simulation(folder / "system-cpu/full-system/simulation.log")
    timing_path = folder / "fpga/reports/timing.tsv"
    timing = dict(line.split("\t") for line in timing_path.read_text().splitlines()[1:]) if timing_path.exists() else {}
    item = dict(status=state, offload=offload, cpu=cpu, timing=timing)
    clocks_path = folder / "fpga/reports/system_clock.tsv"
    clocks = {}
    if clocks_path.exists():
        for line in clocks_path.read_text().splitlines()[1:]:
            connection, clock, period = line.split("\t")
            clocks[connection] = {"clock": clock, "period_ns": float(period)}
    item["clocks"] = clocks
    check_path = folder / "fpga/reports/check_timing.rpt"
    if check_path.exists():
        item["timing_check_counts"] = {
            match[1]: int(match[2]) for match in re.finditer(
                r"^\d+\. checking (\w+) \((\d+)\)", check_path.read_text(), re.M)}
    drc_path = folder / "fpga/reports/drc.rpt"
    if drc_path.exists():
        item["drc_summary"] = {
            match[1].strip(): {"severity": match[2].strip(), "count": int(match[3])}
            for match in re.finditer(
                r"^\| ([\w-]+)\s*\|\s*([^|]+)\|[^|]+\|\s*(\d+)\s*\|",
                drc_path.read_text(), re.M)}
    hierarchy = folder / "fpga/reports/hierarchy.rpt"
    if hierarchy.exists():
        for line in hierarchy.read_text().splitlines():
            fields = [value.strip() for value in line.split("|")]
            if len(fields) != 12 or fields[1].startswith("("):
                continue
            label = ("soc_utilization" if fields[1] == "top" else
                     "accelerator_and_adapter_utilization" if fields[2] == "sqr_bambu_basic_dma64" else None)
            if label:
                item[label] = {key: int(fields[index]) for key, index in
                               (("lut", 3), ("ff", 7), ("ramb36", 8), ("ramb18", 9), ("dsp", 10))}
    deploy = folder / "deploy"
    if deploy.exists():
        checksums = {path.name: digest(path) for path in sorted(deploy.iterdir())
                     if path.is_file() and path.name != "SHA256SUMS"}
        (deploy / "SHA256SUMS").write_text("".join(f"{value}  {key}\n" for key, value in checksums.items()))
        item["artifacts"] = checksums
    # Verify the generated RTL and initialization files were copied verbatim.
    manifest_path = folder / "esp/integration.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for rel, expected in manifest["rtl_sha256"].items():
            assert digest(folder / "esp" / rel) == expected, f"Modified integration input: {name}/{rel}"
            leaf = Path(rel).name
            if leaf in ("forward_kernel.v", "panda_libtech.v") or leaf.endswith(".mem"):
                assert digest(Path(manifest["generated"]) / "bambu" / leaf) == expected
        item["rtl_unchanged"] = True
        template = Path(manifest["simulation_soc_template"])
        soc = Path(manifest["soc"])
        for directory in ("socgen/esp", "socgen/grlib", "socketgen"):
            for source in (template / directory).glob("*.vhd"):
                assert digest(source) == digest(soc / directory / source.name), \
                    f"SoC differs from simulation fixture: {name}/{directory}/{source.name}"
        assert digest(template / "top.vhd") == digest(soc / "top.vhd")
        item["soc_matches_simulation_fixture"] = True
        project = soc / "vivado/esp-xilinx-vc707-xc7vx485t.xpr"
        if project.exists():
            files = {Path(node.attrib["Path"].replace("$PPRDIR", str(project.parent))).resolve()
                     for node in ET.parse(project).findall(".//File")}
            for rel in manifest["rtl_sha256"]:
                if Path(rel).suffix in (".v", ".sv"):
                    assert (folder / "esp" / rel).resolve() in files, \
                        f"Vivado project omits generated input: {name}/{rel}"
            item["vivado_uses_installed_rtl"] = True
    ready = (offload.get("pass") and cpu.get("pass") and state.get("fpga") == "PASS")
    item["ready_for_board"] = bool(ready)
    if a.require_complete:
        assert ready, f"Incomplete campaign: {name}"
        assert item.get("rtl_unchanged") and item.get("soc_matches_simulation_fixture")
        assert item.get("vivado_uses_installed_rtl")
        for result in (offload, cpu):
            assert result["error_e9"] <= result["tolerance_e9"]
        assert "kernel" in offload["regions"] and "kernel" in cpu["regions"]
        assert float(timing["STATS.WNS"]) >= 0 and float(timing["STATS.WHS"]) >= 0
        for connection, expected in (("chip_refclk", 20), ("clkm", 10),
                                     ("accelerator_sequential_cells", 20)):
            assert abs(clocks[connection]["period_ns"] - expected) < 0.001
        assert all(row["severity"] in ("Warning", "Advisory")
                   for row in item["drc_summary"].values()), f"DRC error: {name}"
        for suffix in (".exe", "-cpu.exe", ".bit"):
            assert (deploy / (name + suffix)).stat().st_size > 0
        assert digest(deploy / f"{name}.exe") == digest(folder / "system-esp/full-system/gemm_app.exe")
        assert digest(deploy / f"{name}-cpu.exe") == digest(folder / "system-cpu/full-system/gemm_app.exe")
        assert digest(deploy / f"{name}.bit") == digest(folder / "fpga/top.bit") == digest(soc / "top.bit")
    results[name] = item
    row = dict(kernel=name, system_esp=offload.get("status", "PENDING"),
               system_cpu=cpu.get("status", "PENDING"), fpga=state.get("fpga", "PENDING"),
               error_e9=offload.get("error_e9", ""), cpu_error_e9=cpu.get("error_e9", ""),
               cpu_cycles=cpu.get("regions", {}).get("kernel", ""),
               offload_cycles=offload.get("regions", {}).get("kernel", ""),
               hardware_cycles=offload.get("hardware_cycles", ""))
    row.update({region + "_cycles": offload.get("regions", {}).get(region, "")
                for region in ("alloc", "pack", "accel", "unpack")})
    row.update(wns_ns=timing.get("STATS.WNS", ""), whs_ns=timing.get("STATS.WHS", ""))
    rows.append(row)
(root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
with (root / "results.csv").open("w") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{sum(item['ready_for_board'] for item in results.values())}/{len(results)} ready for board")
print(root / "results.csv")
