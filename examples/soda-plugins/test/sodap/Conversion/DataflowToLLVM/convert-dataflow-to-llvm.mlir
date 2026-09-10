// Passing ConvertDataflowToLLVM its defaults explicitly must change nothing.
// %sodap_bambu_clangxx and %sodap_bambu_include are the values compiled into
// sodap/Config.h, so both runs must produce the same IR and the same
// specialization unit.
//
// REQUIRES: panda

// Each run drops ac_channel_specializations.{cpp,ll} in its working directory,
// so each gets its own.
// RUN: rm -rf %t && mkdir -p %t/default %t/explicit
// RUN: cd %t/default && mlir-opt %S/Inputs/gemm_small.mlir \
// RUN:     --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm)" \
// RUN:     -o %t/default.mlir
// RUN: cd %t/explicit && mlir-opt %S/Inputs/gemm_small.mlir \
// RUN:     --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm{clangxx=%sodap_bambu_clangxx include-panda-path=%sodap_bambu_include})" \
// RUN:     -o %t/explicit.mlir
// RUN: diff %t/default.mlir %t/explicit.mlir
// RUN: diff %t/default/ac_channel_specializations.cpp %t/explicit/ac_channel_specializations.cpp
// RUN: diff %t/default/ac_channel_specializations.ll %t/explicit/ac_channel_specializations.ll
// RUN: FileCheck --implicit-check-not=dataflow.stream %s < %t/explicit.mlir
// RUN: FileCheck --check-prefix=CPP %s < %t/explicit/ac_channel_specializations.cpp

// The specialization unit Clang compiles: one ac_channel<T> per distinct
// element type, with an anchor probe for the layout and one extern "C" probe
// per operation.
// CPP:      #include <ac_channel.h>
// CPP-NEXT: #include <cstdint>
// CPP-NEXT: #include <new>
// CPP:      using C_f32 = ac_channel<float>;
// CPP-NEXT: extern "C" void __df_anchor_f32() { C_f32 c; (void)c; }
// CPP-NEXT: extern "C" __attribute__((always_inline)) void __df_ctor_f32(C_f32 *p) { new (p) C_f32(); }
// CPP-NEXT: extern "C" __attribute__((always_inline)) void __df_dtor_f32(C_f32 *p) { p->~C_f32(); }
// CPP-NEXT: extern "C" __attribute__((always_inline)) float __df_read_f32(C_f32 *p) { return p->read(); }
// CPP-NEXT: extern "C" __attribute__((always_inline)) void __df_write_f32(C_f32 *p, float v) { p->write(v); }
// The design has two f32 channels, but one specialization serves both.
// CPP-NOT:  using C_

// One declaration per entry point, typed as Clang compiled it.
// CHECK-DAG: llvm.func @__df_ctor_f32(!llvm.ptr)
// CHECK-DAG: llvm.func @__df_dtor_f32(!llvm.ptr)
// CHECK-DAG: llvm.func @__df_read_f32(!llvm.ptr) -> f32
// CHECK-DAG: llvm.func @__df_write_f32(!llvm.ptr, f32)

// A stream argument becomes a pointer, and every pointer argument is noalias;
// reads and writes become calls to the probes.
// CHECK-LABEL: func.func @node0(%{{.*}}: !llvm.ptr {llvm.noalias}, %{{.*}}: memref<8x4xf32> {llvm.noalias})
// CHECK:         llvm.call @__df_read_f32(%{{.*}}) : (!llvm.ptr) -> f32

// CHECK-LABEL: func.func @node1(
// CHECK:         llvm.call @__df_read_f32(
// CHECK:         llvm.call @__df_write_f32(%{{.*}}, %{{.*}}) : (!llvm.ptr, f32) -> ()

// The top owns the channels: an alloca of Clang's ac_channel<float> and a
// constructor call for each, the destructors right before the return.
// CHECK-LABEL: func.func @forward(
// CHECK:         [[C0:%.+]] = llvm.alloca %{{.*}} x !llvm.struct<"class.ac_channel"
// CHECK-NEXT:    llvm.call @__df_ctor_f32([[C0]])
// CHECK:         [[C1:%.+]] = llvm.alloca %{{.*}} x !llvm.struct<"class.ac_channel"
// CHECK-NEXT:    llvm.call @__df_ctor_f32([[C1]])
// CHECK:         llvm.call @__df_dtor_f32([[C0]])
// CHECK-NEXT:    llvm.call @__df_dtor_f32([[C1]])
// CHECK-NEXT:    return
