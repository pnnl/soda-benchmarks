module {
    func.func @forward(%input: memref<1x120x16xf32>, %weight: memref<1x1x1xf32>) {
        %output = memref.alloc() {alignment = 64 : i64} : memref<1x120x16xi1>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%input, %weight : memref<1x120x16xf32>, memref<1x1x1xf32>) outs(%output : memref<1x120x16xi1>) {
        ^bb0(%in: f32, %in_36: f32, %out: i1):
          %10 = arith.cmpf oge, %in, %in_36 : f32
          linalg.yield %10 : i1
        }
        return
    }
}