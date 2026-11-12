add_mlir_dialect_library(MLIRSODAPDataflowTransforms
        CopyToDMA.cpp
        LoadStore.cpp
        MergeTasks.cpp
        OutstandingLoad.cpp
        PingPong.cpp
        Schedule.cpp
        ReportLCSCost.cpp
        InferTopInterface.cpp
        NodesToFunc.cpp
        TasksToNodes.cpp
        TilesToLLVM.cpp

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
        MLIRAffineDialect
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRDialectUtils
        MLIRTransforms
        )
