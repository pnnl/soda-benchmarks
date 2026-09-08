add_mlir_dialect_library(MLIRSODAPDataflow
        Dataflow.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap/Dialect/Dataflow

        DEPENDS
        MLIRDataflowIncGen
        SODAPDataflowAttributesIncGen
        SODAPDataflowInterfacesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRAffineDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRSideEffectInterfaces
        MLIRAnalysis
        )
add_subdirectory(Transforms)
