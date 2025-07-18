module {
    func.func @forward(%input: memref<1x1x960xf32>, %weight: memref<1x960x16xf32>, %output: memref<1x1x16xf32>) {
        linalg.batch_matmul ins(%input, %weight : memref<1x1x960xf32>, memref<1x960x16xf32>) outs(%output : memref<1x1x16xf32>)
        return
    }
}