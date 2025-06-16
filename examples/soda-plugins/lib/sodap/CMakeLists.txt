add_mlir_dialect_library(MLIRSODAP
        # SODAPTypes.cpp
        # SODAPDialect.cpp
        # SODAPOps.cpp
        SODAPPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap

        DEPENDS
        # TODO: Removing custom ops and types for now
        # MLIRSODAPOpsIncGen
        MLIRSODAPPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
