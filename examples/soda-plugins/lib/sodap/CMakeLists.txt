add_mlir_dialect_library(MLIRSODAP
        # SODAPTypes.cpp
        # SODAPDialect.cpp
        # SODAPOps.cpp
        MyExtension.cpp
        ViewOpGraph.cpp
        SODAPPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap

        DEPENDS
        # TODO: Removing custom ops and types for now
        # MLIRSODAPOpsIncGen
        MyExtensionCh2IncGen
        MLIRSODAPPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
