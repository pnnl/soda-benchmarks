// RUN: rm -rf %t && mkdir -p %t && cd %t && mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-tiles-to-llvm)" | FileCheck %s
// RUN: FileCheck --check-prefix=LL %s < %t/tiles/k/acc.ll

// The tile's LLVM IR goes to its directory; the module keeps its kernel as a
// declaration, which the launch still resolves against, and nothing else.

// LL: define void @k_node0(ptr %0)
// LL: define void @k(ptr %0)
// LL-NEXT: call void @k_node0(ptr %0)

// CHECK:      module @k attributes {sodap.tile = 0 : i64, sodap.tile_dir = "tiles/k"} {
// CHECK-NEXT:   llvm.func @k(!llvm.ptr)
// CHECK-NEXT: }
// CHECK:      dataflow.launch @k::@k
module @k attributes {sodap.tile = 0 : i64, sodap.tile_dir = "tiles/k"} {
  llvm.func @k_node0(%p: !llvm.ptr) {
    llvm.return
  }
  llvm.func @k(%p: !llvm.ptr) {
    llvm.call @k_node0(%p) : (!llvm.ptr) -> ()
    llvm.return
  }
}
func.func @main(%a: memref<4xf32>) {
  dataflow.launch @k::@k(%a) : (memref<4xf32>) -> ()
  return
}
