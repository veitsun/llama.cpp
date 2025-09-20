
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=debug

cmake --build build --config Release