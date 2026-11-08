add_mlir_library(MLIRSODAPTosa
        TosaTransposes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap/Tosa

        DEPENDS
        SODAPTosaPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRFuncDialect
        MLIRTosaDialect
        MLIRTransformUtils
        )
