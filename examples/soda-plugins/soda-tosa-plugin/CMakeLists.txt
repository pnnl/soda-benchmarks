add_llvm_library(SODATosaPlugin
        MODULE BUILDTREE_ONLY
        soda-tosa-plugin.cpp

        DEPENDS
        MLIRSODAPTosa
        PLUGIN_TOOL
        mlir-opt
        )

# As SODAPlugin (see soda-plugins/CMakeLists.txt): only this project's
# archive, every MLIR symbol bound to mlir-opt's copy at load time.
target_link_libraries(SODATosaPlugin PRIVATE
        -Wl,--whole-archive $<TARGET_FILE:MLIRSODAPTosa> -Wl,--no-whole-archive)

target_include_directories(SODATosaPlugin PRIVATE "${SODAP_BINARY_DIR}/include")
