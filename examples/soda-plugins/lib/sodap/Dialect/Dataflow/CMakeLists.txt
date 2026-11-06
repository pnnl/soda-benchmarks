add_mlir_dialect_library(MLIRSODAPDataflow
        BufferizableOpInterfaceImpl.cpp
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
        MLIRBufferizationDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRSideEffectInterfaces
        MLIRDestinationStyleOpInterface
        MLIRLoopLikeInterface
        MLIRTensorDialect
        MLIRAnalysis
        MLIRTransformUtils
        )
add_subdirectory(Transforms)
