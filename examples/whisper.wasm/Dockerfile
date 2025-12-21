FROM emscripten/emsdk:3.1.74 AS builder

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/ggerganov/whisper.cpp.git

WORKDIR /workspace/whisper.cpp
RUN git checkout v1.7.5
COPY CMakeLists.txt /workspace/whisper.cpp/examples/whisper.wasm
COPY emscripten.cpp /workspace/whisper.cpp/examples/whisper.wasm

RUN mkdir -p build-em
WORKDIR /workspace/whisper.cpp/build-em

RUN emcmake cmake ..
RUN make -j

RUN sed -i 's/libmain.js/main.js/g' /workspace/whisper.cpp/build-em/bin/whisper.wasm/main.js

FROM scratch AS exportor

COPY --from=builder /workspace/whisper.cpp/build-em/bin /dist