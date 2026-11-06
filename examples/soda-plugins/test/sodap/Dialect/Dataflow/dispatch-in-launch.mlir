// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --allow-unregistered-dialect | FileCheck %s

// soda-opt's outlining puts a function's body in a soda.launch, then moves it
// into a soda.func, the kernel: a dispatch is allowed in both. The soda
// dialect isn't in the plugin, so its ops are unregistered here.
// CHECK-LABEL: func.func @in_launch
// CHECK:         "soda.launch"
// CHECK:           dataflow.dispatch
func.func @in_launch() {
  "soda.launch"() ({
    dataflow.dispatch {
      dataflow.yield
    }
    "soda.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: "soda.func"
// CHECK:         dataflow.dispatch
"soda.func"() ({
  dataflow.dispatch {
    dataflow.yield
  }
  "soda.return"() : () -> ()
}) {sym_name = "kernel"} : () -> ()
