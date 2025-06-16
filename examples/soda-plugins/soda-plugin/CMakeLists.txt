add_llvm_library(SODAPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        soda-plugin.cpp

        DEPENDS
        MLIRSODA
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRSODA
        )
target_include_directories(
        SODAPlugin
        PRIVATE
        "${SODAP_BINARY_DIR}/include"
)
