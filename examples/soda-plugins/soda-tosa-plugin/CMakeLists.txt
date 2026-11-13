add_llvm_library(SODATosaPlugin
        MODULE BUILDTREE_ONLY
        soda-tosa-plugin.cpp

        DEPENDS
        MLIRSODAPTosa
        MLIRSODAPDataflow
        PLUGIN_TOOL
        mlir-opt
        )

# As SODAPlugin (see soda-plugins/CMakeLists.txt): only this project's
# archives, every MLIR symbol bound to mlir-opt's copy at load time. The
# Dataflow dialect is for the launch of sodap-tosa-outline-kernel: never load
# this plugin with SODAPlugin, which has its own copy.
target_link_libraries(SODATosaPlugin PRIVATE
        -Wl,--whole-archive $<TARGET_FILE:MLIRSODAPTosa>
        $<TARGET_FILE:MLIRSODAPDataflow> -Wl,--no-whole-archive)

target_include_directories(SODATosaPlugin PRIVATE "${SODAP_BINARY_DIR}/include")
