module {
    func.func @forward(%input: memref<1x1x245x1xf32>, %filter: memref<16x1x7x1xf32>, %output: memref<1x1x120x16xf32>) {
        linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}
            ins(%input, %filter : memref<1x1x245x1xf32>, memref<16x1x7x1xf32>)
            outs(%output : memref<1x1x120x16xf32>)
        return
    }
}