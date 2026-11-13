add_mlir_library(MLIRSODAPTosa
        JoinMultipleTiles.cpp
        TosaTransposes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap/Tosa

        DEPENDS
        SODAPTosaPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRFuncDialect
        MLIRTensorDialect
        MLIRTosaDialect
        MLIRTosaTransforms
        MLIRSODAPDataflow
        MLIRTransformUtils
        )
