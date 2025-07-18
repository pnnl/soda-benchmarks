module {
    func.func @forward(%input: memref<1x1x61x32xf32>, %filter: memref<32x1x3x32xf32>, %output: memref<1x1x30x32xf32>) {
        linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}
            ins(%input, %filter : memref<1x1x61x32xf32>, memref<32x1x3x32xf32>)
            outs(%output : memref<1x1x30x32xf32>)
        return
    }
}