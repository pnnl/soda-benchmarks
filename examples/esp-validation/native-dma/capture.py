#!/usr/bin/env python3
"""Validate generated host + serial tile numerically and capture RTL vectors."""
import ctypes, hashlib, json, math, os, re, subprocess, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve()
fixture=Path(sys.argv[2]).resolve()
repo=Path(__file__).resolve().parents[3]
out=root/'capture';out.mkdir(exist_ok=True)
serial=root/'nonoverlap';host=root/'overlap'
plugin=os.environ['MLIR_PLUGIN_PATH']
flags=['--load-dialect-plugin='+plugin,'--load-pass-plugin='+plugin]
def run(args): subprocess.run(list(map(str,args)),check=True)
lines=(serial/'dataflow.mlir').read_text().splitlines()
start=next(i for i,s in enumerate(lines) if s.startswith('  module @forward_kernel'))
end=next(i for i in range(start+1,len(lines)) if lines[i]=='  }')
kernel='\n'.join(s for s in lines[:start] if s.startswith('#'))+'\nmodule {\n'+'\n'.join(s[2:] for s in lines[start+1:end])+'\n}\n'
kernel=kernel.replace('@forward_kernel','@reference_tile')
(out/'tile.mlir').write_text(kernel)
signature=re.search(r'func.func @reference_tile\((.*?)\) \{',kernel)[1]
shapes=re.findall(r'memref<((?:\d+x)*)f32>',signature)
counts=[math.prod(int(n) for n in s.split('x') if n) for s in shapes]
assert counts and len(counts)==signature.count('%arg')
(out/'tile_signature.h').write_text('#include <cstddef>\nextern "C" void reference_tile('+','.join('float *' for _ in counts)+');\n'+
 'static const size_t tile_sizes[]={'+','.join(map(str,counts))+'};\n'+
 '#define CALL_REFERENCE_TILE(args) reference_tile('+','.join(f'args[{i}].data()' for i in range(len(counts)))+')\n')
run(['mlir-opt',out/'tile.mlir',*flags,'--pass-pipeline=builtin.module(func.func(sodap-dataflow-tasks-to-nodes),sodap-dataflow-nodes-to-func,func.func(convert-linalg-to-loops),canonicalize)','-o',out/'tile_funcs.mlir'])
run(['soda-opt',out/'tile_funcs.mlir','--lower-all-to-llvm=use-bare-ptr-memref-call-conv','--convert-func-to-llvm','--reconcile-unrealized-casts','-o',out/'tile_llvm.mlir'])
run(['mlir-translate','--mlir-to-llvmir',out/'tile_llvm.mlir','-o',out/'tile.ll'])
run(['/usr/bin/clang-19','-O2','-fPIC','-c',repo/'scripts/lib/memref_copy.c','-o',out/'memref_copy.o'])
run(['/usr/bin/clang++-19','-O2','-ffp-contract=off','-fPIC','-shared',host/'05_kernel_riscv.ll',out/'tile.ll',
     out/'memref_copy.o',Path(__file__).with_name('capture_runtime.cpp'),
     '-I'+str(out),'-I'+str(repo/'examples/soda-plugins/include'),'-o',out/'capture.so'])
spec=json.loads((fixture/'fixture.json').read_text())
arrays=[np.fromfile(fixture/a['file'],dtype='<f4').copy() for a in spec['arguments']]
arrays.append(np.full(spec['outputs'],np.nan,dtype=np.float32))
original=[x.copy() for x in arrays[:-1]]
lib=ctypes.CDLL(str(out/'capture.so'));function=lib.forward_kernel
order=list(map(int,(host/'argument_order.txt').read_text().split()))
function.argtypes=[ctypes.c_void_p]*len(order);function.restype=None
function(*[ctypes.c_void_p(arrays[i].ctypes.data) for i in order])
gold=np.fromfile(fixture/'reference.bin',dtype='<f4')
diff=np.abs(arrays[-1].astype(np.float64)-gold.astype(np.float64))
bad=~np.isfinite(arrays[-1])|(diff>spec['atol']+spec['rtol']*np.abs(gold.astype(np.float64)))
assert all(np.array_equal(x,y) for x,y in zip(original,arrays[:-1])), 'Input overwritten'
lib.capture_vectors.argtypes=[ctypes.c_char_p];lib.capture_vectors(str(out).encode())
record={'status':'PASS' if not bad.any() else 'FAIL','errors':int(bad.sum()),'outputs':len(gold),
        'max_absolute_error':float(diff.max()),'scope':'Actual host LLVM and serial scheduled tile executed natively; not RTL',
        'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [host/'05_kernel_riscv.ll',serial/'dataflow.mlir',fixture/'reference.bin',out/'tile.ll']}}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(record,flush=True)
assert record['status']=='PASS'
