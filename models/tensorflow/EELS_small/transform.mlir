module @transforms attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      // transform.foreach_match in %arg0 @match_bitwidth -> @print_bitwidth : (!transform.any_op) -> !transform.any_op
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.collect_matching @match_tile_matmul in %0 : (!transform.any_op) -> !transform.any_op
      transform.include @tiling failures(propagate) (%1) : (!transform.any_op) -> ()
      transform.foreach_match in %0 @match_conv -> @tile_conv : (!transform.any_op) -> !transform.any_op
      %3 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // transform.debug.emit_remark_at %3, "After tiling func" : !transform.any_op
      %lowered = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %3 : (!transform.any_op) -> !transform.any_op
      %4 = transform.collect_matching @match_arit in %lowered : (!transform.any_op) -> !transform.any_op
      transform.foreach %4 : !transform.any_op {
        ^bb0(%arg1: !transform.any_op):
          %parent = transform.get_parent_op %arg1 {nth_parent=1,op_name="scf.for"}  : (!transform.any_op) -> !transform.any_op
          transform.foreach_match in %parent @match_affinefors -> @fullunroll : (!transform.any_op) -> !transform.any_op
      }
      %SROA = transform.apply_registered_pass "affine-scalrep" to %lowered : (!transform.any_op) -> !transform.any_op

      // %1 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %0 : (!transform.any_op) -> !transform.any_op
      // %dim1 = transform.collect_matching @match_dim1 in %arg0 : (!transform.any_op) -> !transform.param<i64>
      // %dim2 = transform.collect_matching @match_dim2 in %arg0 : (!transform.any_op) -> !transform.param<i64>
      // %dim3 = transform.collect_matching @match_dim3 in %arg0 : (!transform.any_op) -> !transform.param<i64>
      // %dim4 = transform.collect_matching @match_dim4 in %arg0 : (!transform.any_op) -> !transform.param<i64>
      // %tiled_conv, %loops:7 = transform.structured.tile_using_for %1 tile_sizes [1,1,1,%dim1,%dim2,%dim3,%dim4] : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

      // transform.debug.emit_param_as_remark %dim1, "dim1:" : !transform.param<i64>
      // transform.debug.emit_param_as_remark %dim2, "dim2:" : !transform.param<i64>
      // transform.debug.emit_param_as_remark %dim3, "dim3:" : !transform.param<i64>
      // transform.debug.emit_param_as_remark %dim4, "dim4:" : !transform.param<i64>
    //   %conv = transform.collect_matching @match_conv in %arg0 : (!transform.any_op) -> !transform.any_op
    //   %1 = transform.foreach_match in %arg0 @match_dimension_capture -> @do_nothing : (!transform.any_op) -> !transform.any_op
    //   transform.include @tile_conv failures(propagate) (%1) : (!transform.any_op) -> ()
    //   transform.include @tiling failures(propagate) (%1) : (!transform.any_op) -> ()
    //   %lowered = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %0 : (!transform.any_op) -> !transform.any_op
    //   %4 = transform.collect_matching @match_arit in %lowered : (!transform.any_op) -> !transform.any_op
    //   transform.foreach %4 : !transform.any_op {
        // ^bb0(%arg1: !transform.any_op):
        //   %parent = transform.get_parent_op %arg1 {nth_parent=1,op_name="scf.for"}  : (!transform.any_op) -> !transform.any_op
        //   transform.foreach_match in %parent @match_affinefors -> @fullunroll : (!transform.any_op) -> !transform.any_op
    //   }
    //   %SROA = transform.apply_registered_pass "affine-scalrep" to %lowered : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }

    transform.named_sequence @match_conv(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.match.operation_name %arg0 ["linalg.conv_2d_nhwc_fhwc"] : !transform.any_op
      transform.yield %arg0 : !transform.any_op
    }

    transform.named_sequence @tile_conv(%arg0: !transform.any_op {transform.consumed}) { // %s:4 = transform.split_handle %arg1 : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      %dim1 = transform.collect_matching @match_dim1 in %arg0 : (!transform.any_op) -> !transform.param<i64>
      %dim2 = transform.collect_matching @match_dim2 in %arg0 : (!transform.any_op) -> !transform.param<i64>
      %dim3 = transform.collect_matching @match_dim3 in %arg0 : (!transform.any_op) -> !transform.param<i64>
      %dim4 = transform.collect_matching @match_dim4 in %arg0 : (!transform.any_op) -> !transform.param<i64>
      // transform.debug.emit_param_as_remark %dim1, "dim1:" : !transform.param<i64>
      // transform.debug.emit_param_as_remark %dim2, "dim2:" : !transform.param<i64>
      // transform.debug.emit_param_as_remark %dim3, "dim3:" : !transform.param<i64>
      // transform.debug.emit_param_as_remark %dim4, "dim4:" : !transform.param<i64>
      %tiled_conv, %loops:7 = transform.structured.tile_using_for %arg0 tile_sizes [1,1,1,%dim1,%dim2,%dim3,%dim4] : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }

    transform.named_sequence @match_dim1(%arg0: !transform.any_op {transform.readonly}) -> (!transform.param<i64>) {
    %d = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%arg1: !transform.any_op):
      transform.match.operation_name %arg0 ["linalg.conv_2d_nhwc_fhwc"] : !transform.any_op
      %dim = transform.match.structured.dim %arg1[3] : (!transform.any_op) -> !transform.param<i64>
    //   %s:4 = transform.split_handle %dims : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    //   %dim:4 = transform.split_handle %dims : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      transform.match.structured.yield %dim : !transform.param<i64>
    }
    transform.yield %d : !transform.param<i64>
    }

    transform.named_sequence @match_dim2(%arg0: !transform.any_op {transform.readonly}) -> (!transform.param<i64>) {
    %d = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%arg1: !transform.any_op):
      transform.match.operation_name %arg0 ["linalg.conv_2d_nhwc_fhwc"] : !transform.any_op
      %dim = transform.match.structured.dim %arg1[4] : (!transform.any_op) -> !transform.param<i64>
    //   %s:4 = transform.split_handle %dims : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    //   %dim:4 = transform.split_handle %dims : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      transform.match.structured.yield %dim : !transform.param<i64>
    }
    transform.yield %d : !transform.param<i64>
    }

    transform.named_sequence @match_dim3(%arg0: !transform.any_op {transform.readonly}) -> (!transform.param<i64>) {
    %d = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%arg1: !transform.any_op):
      transform.match.operation_name %arg0 ["linalg.conv_2d_nhwc_fhwc"] : !transform.any_op
      %dim = transform.match.structured.dim %arg1[5] : (!transform.any_op) -> !transform.param<i64>
    //   %s:4 = transform.split_handle %dims : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    //   %dim:4 = transform.split_handle %dims : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      transform.match.structured.yield %dim : !transform.param<i64>
    }
    transform.yield %d : !transform.param<i64>
    }

    transform.named_sequence @match_dim4(%arg0: !transform.any_op {transform.readonly}) -> (!transform.param<i64>) {
    %d = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%arg1: !transform.any_op):
      transform.match.operation_name %arg0 ["linalg.conv_2d_nhwc_fhwc"] : !transform.any_op
      %dim = transform.match.structured.dim %arg1[6] : (!transform.any_op) -> !transform.param<i64>
    //   %s:4 = transform.split_handle %dims : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    //   %dim:4 = transform.split_handle %dims : (!transform.param<i64>) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      transform.match.structured.yield %dim : !transform.param<i64>
    }
    transform.yield %d : !transform.param<i64>
    }

    transform.named_sequence @match_tile_matmul(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
     transform.match.structured %arg0 : !transform.any_op {
     ^bb0(%arg1: !transform.any_op):
       transform.match.operation_name %arg0 ["linalg.batch_matmul"] : !transform.any_op
     }
     transform.yield %arg0 : !transform.any_op
    }

    transform.named_sequence @tiling(%arg0: !transform.any_op {transform.consumed}) {
      %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %arg0 tile_sizes [1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield 
    }

    transform.named_sequence @match_arit(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.match.operation_name %arg0 ["arith.addf"] : !transform.any_op
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @match_affinefors(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.match.operation_name %arg0 ["affine.for"] : !transform.any_op
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @fullunroll(%arg0: !transform.any_op {transform.consumed}) {
      transform.loop.fullunroll %arg0 : !transform.any_op
      transform.yield 
    }
  }