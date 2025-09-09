This requires compilation of the mlir-opt plugin tool.



# Questions

Number of resource uses should be 1:

soda-benchmarks/examples/soda-plugins/examples/instrumentation/assertLessThen/IP_integration/constraints_STD.xml



```mlir
module {
  // CHECK-LABEL: func.func @example
  func.func @example(%arg0: memref<8x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<8x8xf32>) -> memref<8x8xf32> {
    // Start a HW counter to count cycles globaly
    // CHECK: @sodaHWC(start, loc0) : (!signal, !location)
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c8 step %c1 {
      // Start a HW counter to count cycles for this loop
      // CHECK: @sodaHWC(start, loc1) : (!signal, !location)
      scf.for %arg4 = %c0 to %c8 step %c1 {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<8x16xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<16x8xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<8x8xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<8x8xf32>
        }
      }
      // Stop the HW counter for this loop
      // CHECK: @sodaHWC(stop, loc1) : (!signal, !location)
    }
    return %arg2 : memref<8x8xf32>
    // Stop the global HW counter
    // CHECK: @sodaHWC(stop, loc0) : (!signal, !location)
  }
}
```

Return values in the integrated function?

## Limitations

Is it possible to bind commands to the same module without affecting all the bindings?

- If integrated IP are in different scopes we need function proxy


## Example of IP in bambu

- IP with return port: 
  - panda/etc/lib/technology/C_HLS_IPs.xml line 4640 - check return_port


# TODO

- [] Use function proxy
- [] Check bambu repo for IP integration tutorials
- [] Vito's idea: replace existing llvm instruciton with call that maps into instrumented verilog of that function.
- [] Assertion modules could be not clocked
- [] HW counters should include clock and memory

Examples
- [] Non-clocked non-memory interface
- [x] clocked non-memory interface
- [] clocked memory interface

Non instrumentation examples
- --generate-interface-infer - paramenters will follow a protocol - for example, burst transactions
  - We could integrate this kind of burst IP to copy from func interface to alloca buffers.


Omer:
- []: Verify the code that generates the IP - where is this code?


## OBS:

Looking at bambu State graph