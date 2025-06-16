# add_mlir_dialect(SODAPOps sodap)
# add_mlir_doc(SODAPDialect SODAPDialect sodap/ -gen-dialect-doc)
# add_mlir_doc(SODAPOps SODAPOps sodap/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS SodaPPasses.td)
mlir_tablegen(SodaPPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRSodaPPassesIncGen)
