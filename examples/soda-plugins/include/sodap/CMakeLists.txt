# add_mlir_dialect(SODAPOps sodap)
# add_mlir_doc(SODAPDialect SODAPDialect sodap/ -gen-dialect-doc)
# add_mlir_doc(SODAPOps SODAPOps sodap/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS SODAPPasses.td)
mlir_tablegen(SODAPPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRSODAPPassesIncGen)


set(LLVM_TARGET_DEFINITIONS MyExtension.td)
mlir_tablegen(MyExtension.h.inc -gen-op-decls)
mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)
add_public_tablegen_target(MyExtensionCh2IncGen)

set(LLVM_TARGET_DEFINITIONS SODAPTransforms.td)
mlir_tablegen(SODAPTransforms.h.inc -gen-op-decls)
mlir_tablegen(SODAPTransforms.cpp.inc -gen-op-defs)
add_public_tablegen_target(SODAPTransformsIncGen)

set(LLVM_TARGET_DEFINITIONS AnalysisPasses.td)
mlir_tablegen(AnalysisPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(SODAPAnalysisPassesIncGen)