add_mlir_dialect_library(MLIRSODAP
        GenerateLinalgSummary.cpp
        MyExtension.cpp
        ViewOpGraph.cpp
        SODAPPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap

        DEPENDS
        SODAPAnalysisPassesIncGen
        MyExtensionCh2IncGen
        MLIRSODAPPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
