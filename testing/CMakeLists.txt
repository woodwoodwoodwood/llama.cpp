# Kernel micro-benchmark: GSQ2 vs Q2_K (dequant / mmvq / mmq)
#
# 通过 ggml backend API 调用真实 CUDA 内核，纯净测三类 kernel 性能。
# 用法见 bench_kernels.cu 文件头注释。

set(TARGET bench-kernels)

add_executable(${TARGET} bench_kernels.cu)

# 链接 ggml（含 cuda 后端，通过 ggml::ggml alias）+ llama
target_link_libraries(${TARGET} PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include
)
