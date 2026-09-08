add_mlir_dialect_library(MLIRSODAPAffineToDataflow
        ConvertMemRefsToStreams.cpp
        CreateDataflowTasks.cpp
        Pipelines.cpp
        Utils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap/Conversion/AffineToDataflow

        DEPENDS
        SODAPAffineToDataflowPassesIncGen

        LINK_LIBS PUBLIC
        MLIRSODAPDataflow
        MLIRSODAPDataflowTransforms
        MLIRIR
        MLIRPass
        MLIRAffineDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRTransforms
        )
