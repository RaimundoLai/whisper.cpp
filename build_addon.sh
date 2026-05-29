#!/bin/bash
set -e

cd examples/addon.node
npm install
cd ../../
mkdir -p artifacts

echo "Building Metal variant..."
npx cmake-js compile -T addon.node -B Release \
  --CDBUILD_SHARED_LIBS=OFF \
  --CDWHISPER_STATIC=ON \
  --CDGGML_METAL=ON \
  --CDGGML_METAL_USE_BF16=ON \
  --CDGGML_METAL_EMBED_LIBRARY=ON \
  --runtime=electron \
  --runtime-version=39.8.10 \
  --arch=arm64
cp build/Release/addon.node.node artifacts/addon-macos-arm64.node

echo "Clearing build cache..."
rm -rf build/CMakeCache.txt build/CMakeFiles/

echo "Building CoreML variant..."
npx cmake-js compile -T addon.node -B Release \
  --CDBUILD_SHARED_LIBS=OFF \
  --CDWHISPER_STATIC=ON \
  --CDWHISPER_COREML=ON \
  --CDGGML_METAL=ON \
  --CDGGML_METAL_USE_BF16=ON \
  --CDGGML_METAL_EMBED_LIBRARY=ON \
  --runtime=electron \
  --runtime-version=39.8.10 \
  --arch=arm64

cp build/Release/addon.node.node artifacts/addon-macos-arm64-coreml.node

echo "Build complete."
