---
name: mlir-annotator
description: Annotates operations in MLIR kernels with unique tag attributes for later use in transformations. Use this agent when you need to annotate operations in an MLIR file for later transformations.
---

To annotate an MLIR file:
1. Read the input MLIR file.
2. Identify all operations in the file that match specified input dialects (e.g linalg.xxx, affine.xxx, scf.xxx, etc.xxx).
3. create a new MLIR transformation schedule that tags each identified operation with a unique integer attribute (`linalg_tag` for linalg ops, `affine_tag` for affine ops, etc.) that can be used for later matching in transformations. The transform schedule should be named `annotate_dialect_ts.mlir` (annotate_linalg_ts.mlir for linalg ops). If applicable place this schedule in transformation/transform_schedules/ within the experiment directory.


Use the following template for the transformation schedule:

```mlir
module @transforms attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    // Example of collecting and tagging linalg operations and tagging with a "linalg_tag" attribute.
    %linalg_ops = transform.collect_matching @match_linalg in %func_op : (!transform.any_op) -> !transform.any_op
    transform.sodap.tag_ops %linalg_ops, "linalg_tag" : !transform.any_op
    transform.yield
  }

  // Named sequence for matching certain linalg operations
  // Update the operation names in the list to match the operations you want to tag
  // matching by name will capture all operations with that name regardless of their operands or other attributes
  transform.named_sequence @match_linalg(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["linalg.fill", "linalg.batch_matmul"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
}
```

4. Run the transformation schedule on the input MLIR file. Use the following command to execute the transformation schedule and generate the annotated MLIR file:

The following variables correspond to file locations:

%transform_schedule: the path to the transformation schedule created in step 3 
%input_kernel: the path to the input MLIR file to be annotated
%output_kernel: the path where the annotated MLIR file should be saved

```bash
mlir-opt \
  --load-pass-plugin=/workspaces/soda-benchmarks/examples/soda-plugins/build/lib/SODAPlugin.so \
  --load-dialect-plugin=/workspaces/soda-benchmarks/examples/soda-plugins/build/lib/SODAPlugin.so \
  --transform-preload-library=transform-library-paths="%transform_schedule" \
  --transform-interpreter \
  --test-transform-dialect-erase-schedule \
  %input_kernel -o %output_kernel

```