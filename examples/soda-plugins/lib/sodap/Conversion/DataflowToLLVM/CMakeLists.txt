add_mlir_dialect_library(MLIRSODAPDataflowToLLVM
        ConvertDataflowToLLVM.cpp
        EmitBambuArchitecture.cpp
        Pipelines.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap/Conversion/DataflowToLLVM

        DEPENDS
        SODAPDataflowToLLVMPassesIncGen

        LINK_COMPONENTS
        Core
        IRReader
        Support

        LINK_LIBS PUBLIC
        MLIRSODAPDataflow
        MLIRIR
        MLIRPass
        MLIRFuncDialect
        MLIRLLVMDialect
        MLIRTargetLLVMIRImport
        MLIRAffineToStandard
        MLIRBufferizationTransforms
        MLIRControlFlowToLLVM
        MLIRFuncToLLVM
        MLIRIndexToLLVM
        MLIRMathToLLVM
        MLIRMemRefToLLVM
        MLIRMemRefTransforms
        MLIRReconcileUnrealizedCasts
        MLIRSCFToControlFlow
        MLIRTransforms
        )
