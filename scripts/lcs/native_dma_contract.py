#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Record a pipeline's DMA contract; select an adapter without mixing old IR.

This records provenance, not a proof for arbitrary manually modified hardware.
Only tosa_to_lcs.sh emits this contract after successful admitted DMA lowering.
"""
import argparse
import hashlib
import json
from pathlib import Path

FILES = ('05_hw_acc.ll', '05_kernel_riscv.ll', 'architecture.xml', 'layout.txt')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(directory):
    layout = dict(line.split('=', 1) for line in (directory/'layout.txt').read_text().splitlines())
    contract = {
        'version': 1,
        'producer': 'tosa_to_lcs: receiver admission and retained output-tile readiness',
        'dma_word_stride': int(layout['dma_word_stride']),
        'sha256': {name: digest(directory/name) for name in FILES},
    }
    (directory/'native_dma_contract.json').write_text(json.dumps(contract, indent=2)+'\n')

def select(directory, mode):
    repo = Path(__file__).resolve().parents[2]
    if mode == 'buffered':
        return repo/'examples/esp-validation/generated-gemm/sqr_bambu_basic_dma64.sv'
    path = directory/'native_dma_contract.json'
    if not path.exists():
        raise ValueError('Thin wrapper requires fresh tosa_to_lcs output; use --wrapper buffered for legacy artifacts')
    contract = json.loads(path.read_text())
    if contract['version'] != 1 or contract['dma_word_stride'] != 2:
        raise ValueError('Thin 64-bit wrapper requires native DMA contract v1 with dma_word_stride=2')
    for name in FILES:
        if digest(directory/name) != contract['sha256'][name]:
            raise ValueError(f'Stale native DMA contract: {name}')
    for generated, source in (('accelerator.ll','05_hw_acc.ll'), ('architecture.xml','architecture.xml')):
        if digest(directory/'bambu_syn'/generated) != contract['sha256'][source]:
            raise ValueError(f'Stale Bambu synthesis input: {generated}')
    return repo/'examples/esp-validation/native-dma/thin_dma64.sv'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['record', 'select'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--wrapper', choices=['thin', 'buffered'], default='thin')
    args = parser.parse_args()
    try:
        if args.action == 'record':
            record(args.directory)
        else:
            print(select(args.directory, args.wrapper))
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f'{error}\n')

if __name__ == '__main__':
    main()
