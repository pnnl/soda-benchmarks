add_mlir_dialect_library(MLIRLinalgToESPRuntime
        ESPPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap

        DEPENDS
        MLIRSODAPPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRMemRefDialect
        )
