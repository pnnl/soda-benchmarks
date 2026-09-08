add_mlir_dialect_library(MLIRSODAPDataflowTransforms
        NodesToFunc.cpp
        TasksToNodes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap/Dialect/Dataflow/Transforms

        DEPENDS
        SODAPDataflowTransformsPassesIncGen

        LINK_LIBS PUBLIC
        MLIRSODAPDataflow
        MLIRIR
        MLIRPass
        MLIRAnalysis
        MLIRFuncDialect
        MLIRTransforms
        )
