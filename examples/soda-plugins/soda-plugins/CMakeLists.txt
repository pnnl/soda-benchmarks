add_llvm_library(SODAPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        soda-plugins.cpp

        DEPENDS
        MLIRSODAP
        MLIRLinalgToESPRuntime
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRSODAP
        # TODO: We should not need to link against the entire ESPRuntime
        #       Runtime libraries could be split into smaller libraries in the future to avoid this issue.
        MLIRLinalgToESPRuntime
        )
target_include_directories(
        SODAPlugin
        PRIVATE
        "${SODAP_BINARY_DIR}/include"
)
