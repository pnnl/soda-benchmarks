add_mlir_dialect(Dataflow dataflow)

set(LLVM_TARGET_DEFINITIONS Interfaces.td)
mlir_tablegen(DataflowInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(DataflowInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(SODAPDataflowInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Dataflow.td)
mlir_tablegen(DataflowAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(DataflowAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(DataflowEnums.h.inc -gen-enum-decls)
mlir_tablegen(DataflowEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(SODAPDataflowAttributesIncGen)

add_subdirectory(Transforms)
