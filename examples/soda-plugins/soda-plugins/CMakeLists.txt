add_llvm_library(SODAPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        soda-plugins.cpp

        DEPENDS
        MLIRSODAP
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRSODAP
        )
target_include_directories(
        SODAPlugin
        PRIVATE
        "${SODAP_BINARY_DIR}/include"
)
