# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()


if (WHISPER_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (WHISPER_COREML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)

    if (COREML_FRAMEWORK)
        message(STATUS "CoreML framework found")

        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_COREML)
    else()
        message(FATAL_ERROR "CoreML framework not found")
    endif()

    if (WHISPER_COREML_ALLOW_FALLBACK)
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_COREML_ALLOW_FALLBACK)
    endif()
endif()

if (WHISPER_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

#
# libraries
#

# whisper.coreml

if (WHISPER_COREML)
    set(TARGET whisper.coreml)

    add_library(${TARGET}
        coreml/whisper-compat.m
        coreml/whisper-encoder.h
        coreml/whisper-encoder.mm
        coreml/whisper-encoder-impl.h
        coreml/whisper-encoder-impl.m
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC
        .
        )

    target_link_libraries(${TARGET} PRIVATE ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})

    set_target_properties(${TARGET} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        )

    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
    install(TARGETS ${TARGET} LIBRARY)
endif()

if (WHISPER_OPENVINO)
    set(TARGET whisper.openvino)

    add_library(${TARGET} OBJECT
        openvino/whisper-openvino-encoder.h
        openvino/whisper-openvino-encoder.cpp
        )

    target_include_directories(${TARGET} PUBLIC
        .
        )

    set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_OPENVINO)

    target_link_libraries(${TARGET} PRIVATE ggml openvino::runtime)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# sense-voice sources (from SenseVoice.cpp submodule)
set(SENSE_VOICE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/SenseVoice.cpp/sense-voice/csrc)
set(SENSE_VOICE_SOURCES
    ${SENSE_VOICE_DIR}/common.cc
    ${SENSE_VOICE_DIR}/sense-voice.cc
    ${SENSE_VOICE_DIR}/sense-voice-encoder.cc
    ${SENSE_VOICE_DIR}/sense-voice-decoder.cc
    ${SENSE_VOICE_DIR}/sense-voice-frontend.cc
    ${SENSE_VOICE_DIR}/silero-vad.cc
    ${SENSE_VOICE_DIR}/fftsg.cc
)

# qwen3-asr sources (from qwen3-asr-cpp)
set(QWEN3_ASR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/qwen3-asr-cpp)
set(QWEN3_ASR_SOURCES
    ${QWEN3_ASR_DIR}/mel_spectrogram.cpp
    ${QWEN3_ASR_DIR}/gguf_loader.cpp
    ${QWEN3_ASR_DIR}/audio_encoder.cpp
    ${QWEN3_ASR_DIR}/text_decoder.cpp
    ${QWEN3_ASR_DIR}/audio_injection.cpp
    ${QWEN3_ASR_DIR}/qwen3_asr.cpp
    ${QWEN3_ASR_DIR}/forced_aligner.cpp
)

set(CRISPASR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/CrispASR)
set(CRISP_AUDIO_DIR ${CRISPASR_DIR}/crisp_audio)

file(GLOB CRISPASR_SOURCES 
    "${CRISPASR_DIR}/src/*.cpp" 
    "${CRISPASR_DIR}/src/*.c"
    "${CRISPASR_DIR}/src/*.cc"
)
file(GLOB CRISPASR_CORE_SOURCES
    "${CRISPASR_DIR}/src/core/*.cpp"
)
set(CRISPASR_RNNOISE_SOURCES
    "${CRISPASR_DIR}/src/rnnoise/denoise.c"
    "${CRISPASR_DIR}/src/rnnoise/rnn.c"
    "${CRISPASR_DIR}/src/rnnoise/rnn_data.c"
    "${CRISPASR_DIR}/src/rnnoise/pitch.c"
    "${CRISPASR_DIR}/src/rnnoise/celt_lpc.c"
    "${CRISPASR_DIR}/src/rnnoise/kiss_fft.c"
)
set(CRISP_AUDIO_SOURCES
    "${CRISP_AUDIO_DIR}/src/audio_tower.cpp"
)
list(APPEND CRISPASR_SOURCES ${CRISPASR_CORE_SOURCES})
list(APPEND CRISPASR_SOURCES ${CRISPASR_RNNOISE_SOURCES})
list(APPEND CRISPASR_SOURCES ${CRISP_AUDIO_SOURCES})
list(REMOVE_ITEM CRISPASR_SOURCES "${CRISPASR_DIR}/src/crispasr.cpp" "${CRISPASR_DIR}/src/chat.cpp" "${CRISPASR_DIR}/src/wav2vec2-ggml-debug.cpp")

# Add CoreML support for SenseVoice when WHISPER_COREML is enabled
if (WHISPER_COREML)
    set(SENSE_VOICE_COREML_SOURCE ${SENSE_VOICE_DIR}/coreml/sense-voice-encoder.mm)
    list(APPEND SENSE_VOICE_SOURCES ${SENSE_VOICE_COREML_SOURCE})
    set_source_files_properties(${SENSE_VOICE_COREML_SOURCE} PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set(SENSE_VOICE_COREML ON)
    add_compile_definitions(SENSE_VOICE_USE_COREML)
endif()

# whisper

add_library(whisper
            ../include/whisper.h
            whisper-arch.h
            whisper.cpp
            ${SENSE_VOICE_SOURCES}
            ${QWEN3_ASR_SOURCES}
            ${CRISPASR_SOURCES}
            )

add_library(parakeet
            ../include/parakeet.h
            parakeet-arch.h
            parakeet.cpp
            )

target_include_directories(parakeet PUBLIC . ../include)
target_compile_features   (parakeet PUBLIC cxx_std_11)
target_link_libraries(parakeet PUBLIC ggml Threads::Threads)

# Set the version numbers
set_target_properties(whisper PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
)

set_target_properties(parakeet PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
)

target_include_directories(whisper PUBLIC . ../include ${SENSE_VOICE_DIR} ${QWEN3_ASR_DIR} ${CRISPASR_DIR}/include ${CRISPASR_DIR}/src ${CRISP_AUDIO_DIR}/include ../ggml/src)
target_compile_features   (whisper PUBLIC cxx_std_20) # don't bump

if (CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_BIG_ENDIAN)
    set(PARAKEET_EXTRA_FLAGS ${PARAKEET_EXTRA_FLAGS} -DPARAKEET_BIG_ENDIAN)
endif()

if (WHISPER_EXTRA_FLAGS)
    target_compile_options(whisper PRIVATE ${WHISPER_EXTRA_FLAGS})
endif()

if (PARAKEET_EXTRA_FLAGS)
    target_compile_options(parakeet PRIVATE ${PARAKEET_EXTRA_FLAGS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(whisper PUBLIC ggml Threads::Threads)

# Link Accelerate framework on Apple for qwen3-asr mel spectrogram
if(APPLE)
    target_link_libraries(whisper PRIVATE "-framework Accelerate")
endif()

# Add tokenizers-cpp for SenseVoice support
set(TOKENIZERS_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third-party/tokenizers-cpp")
if(EXISTS "${TOKENIZERS_CPP_DIR}/CMakeLists.txt")
    if (NOT TARGET tokenizers_cpp)
        set(CMAKE_POLICY_VERSION_MINIMUM "3.5")
        add_subdirectory("${TOKENIZERS_CPP_DIR}" tokenizers_cpp)
    endif()
    target_link_libraries(whisper PUBLIC tokenizers_cpp)
    target_compile_definitions(whisper PRIVATE HAVE_TOKENIZERS_CPP)
endif()


if (WHISPER_COREML)
    target_link_libraries(whisper PRIVATE whisper.coreml)
    # Ensure frameworks are linked for SenseVoice mm files included in whisper target
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)
    target_link_libraries(whisper PRIVATE ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})
    add_definitions(-DSENSE_VOICE_USE_COREML)
endif()

if (WHISPER_OPENVINO)
    target_link_libraries(whisper PRIVATE whisper.openvino)
endif()

if (WHISPER_MKL)
    target_link_libraries(whisper PRIVATE MKL::MKL)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(whisper PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(whisper PRIVATE WHISPER_SHARED WHISPER_BUILD)

    set_target_properties(parakeet PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(parakeet PRIVATE PARAKEET_SHARED PARAKEET_BUILD)
endif()
