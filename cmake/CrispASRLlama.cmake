# Build CrispASR's vendored llama.cpp sources as a private static library.
# The target intentionally uses the parent project's ggml libraries so an
# addon embedding whisper does not load a second copy of ggml.
function(crispasr_add_llama_core whisper_target crispasr_source_dir)
    set(_crispasr_llama_dir "${crispasr_source_dir}/examples/talk-llama")
    file(GLOB _crispasr_llama_models CONFIGURE_DEPENDS
         "${_crispasr_llama_dir}/models/*.cpp")

    add_library(crispasr-llama-core STATIC
        "${_crispasr_llama_dir}/llama.cpp"
        "${_crispasr_llama_dir}/llama-adapter.cpp"
        "${_crispasr_llama_dir}/llama-arch.cpp"
        "${_crispasr_llama_dir}/llama-batch.cpp"
        "${_crispasr_llama_dir}/llama-chat.cpp"
        "${_crispasr_llama_dir}/llama-context.cpp"
        "${_crispasr_llama_dir}/llama-cparams.cpp"
        "${_crispasr_llama_dir}/llama-grammar.cpp"
        "${_crispasr_llama_dir}/llama-graph.cpp"
        "${_crispasr_llama_dir}/llama-hparams.cpp"
        "${_crispasr_llama_dir}/llama-impl.cpp"
        "${_crispasr_llama_dir}/llama-io.cpp"
        "${_crispasr_llama_dir}/llama-kv-cache.cpp"
        "${_crispasr_llama_dir}/llama-kv-cache-iswa.cpp"
        "${_crispasr_llama_dir}/llama-memory.cpp"
        "${_crispasr_llama_dir}/llama-memory-recurrent.cpp"
        "${_crispasr_llama_dir}/llama-memory-hybrid.cpp"
        "${_crispasr_llama_dir}/llama-memory-hybrid-iswa.cpp"
        "${_crispasr_llama_dir}/llama-mmap.cpp"
        "${_crispasr_llama_dir}/llama-model-loader.cpp"
        "${_crispasr_llama_dir}/llama-model-saver.cpp"
        "${_crispasr_llama_dir}/llama-model.cpp"
        "${_crispasr_llama_dir}/llama-quant.cpp"
        "${_crispasr_llama_dir}/llama-sampler.cpp"
        "${_crispasr_llama_dir}/llama-vocab.cpp"
        "${_crispasr_llama_dir}/unicode.cpp"
        "${_crispasr_llama_dir}/unicode-data.cpp"
        ${_crispasr_llama_models})

    target_include_directories(crispasr-llama-core PRIVATE
        "${_crispasr_llama_dir}")
    target_compile_features(crispasr-llama-core PUBLIC cxx_std_17)
    target_link_libraries(crispasr-llama-core PUBLIC ggml ggml-base Threads::Threads)
    set_target_properties(crispasr-llama-core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        C_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

    if(NOT MSVC)
        target_compile_options(crispasr-llama-core PRIVATE
            -O3
            -Wno-cast-qual
            -Wno-unused-function
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-shadow
            -Wno-missing-declarations
            -Wno-missing-noreturn
            -Wno-pedantic
            -Wno-deprecated-declarations)
    endif()

    if(WIN32)
        target_compile_definitions(crispasr-llama-core PRIVATE _WIN32_WINNT=0x0602)
    endif()

    # crispasr_c_api.cpp and index_echo.cpp need llama.h during compilation.
    # PRIVATE keeps the vendored llama API out of whisper's installed interface.
    target_include_directories(${whisper_target} PRIVATE "${_crispasr_llama_dir}")
    target_link_libraries(${whisper_target} PRIVATE crispasr-llama-core)
endfunction()
