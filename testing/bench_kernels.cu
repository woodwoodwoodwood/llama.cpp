// Micro-benchmark: compare GSQ2 vs Q2_K kernel performance (dequant / mmvq / mmq).
//
// 测三类纯 kernel（通过 ggml backend 调真实生产内核）：
//   1. DEQUANT: ggml_cpy(quant -> F32)            — 触发 dequantize_block 内核
//   2. MMVQ:    ggml_mul_mat(quant, F32 vec)      — batch=1，走 mul_mat_vec_q 内核
//   3. MMQ:     ggml_mul_mat(quant, F32 matrix)   — batch 大，走 mul_mat_q (MMQ) 内核
//
// 编译（在已构建的 build 目录下）：
//   nvcc -O2 -std=c++17 -I../include -I../ggml/include \
//       testing/bench_kernels.cu \
//       -L./bin -lggml -lggml-base -lggml-cpu -lggml-cuda \
//       -Xlinker -rpath -Xlinker ./bin \
//       -lpthread -lcudart -lcublas \
//       -Wno-deprecated-gpu-targets \
//       -o bench_kernels
//
// 用法: ./bench_kernels [n_rows] [n_cols] [mmq_batch] [iters]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

using hrc = std::chrono::high_resolution_clock;
static double now_s() {
    return std::chrono::duration<double>(hrc::now().time_since_epoch()).count();
}

// 量化一行 float -> 指定类型，返回量化 bytes
static std::vector<uint8_t> quantize_row(enum ggml_type type, int64_t n, const float * src) {
    size_t row_size = ggml_row_size(type, n);
    std::vector<uint8_t> dst(row_size, 0);
    ggml_quantize_init(type);
    ggml_quantize_chunk(type, src, dst.data(), 0, 1, n, nullptr);
    return dst;
}

struct Bench {
    ggml_backend_t backend;
    int iters;
};

// ---------- DEQUANT ----------
static void bench_dequant(Bench & B, enum ggml_type type, int64_t n) {
    std::vector<float> src(n);
    for (int64_t i = 0; i < n; i++) src[i] = (float)((rand()%2000)-1000)/500.0f;
    auto qbytes = quantize_row(type, n, src.data());

    struct ggml_init_params p = { .mem_size = 1024*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, type, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_cpy(ctx, a, b);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Allocate all tensors using backend (like test-backend-ops.cpp does)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, B.backend);
    if (buf == NULL) {
        fprintf(stderr, "failed to allocate tensors for dequant\n");
        ggml_free(ctx);
        return;
    }

    // Set input data
    ggml_backend_tensor_set(a, qbytes.data(), 0, qbytes.size());

    // warmup
    ggml_backend_graph_compute(B.backend, gf);

    // timed
    double t0 = now_s();
    for (int i = 0; i < B.iters; i++) {
        ggml_backend_graph_compute(B.backend, gf);
    }
    double sec = (now_s() - t0) / B.iters;

    printf("%-22s %12.6f %14.3f M elem/s\n",
           (std::string(ggml_type_name(type))+" dequant").c_str(), sec, n/sec/1e6);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// ---------- MUL_MAT (mmvq when batch==1, mmq when batch 大) ----------
static void bench_mul_mat(Bench & B, enum ggml_type type, int64_t n_rows, int64_t n_cols, int64_t batch) {
    // 列对齐 block（GSQ2=128, Q2_K=256）
    int64_t blck = ggml_blck_size(type);
    n_cols = (n_cols / blck) * blck;
    if (n_cols < blck) n_cols = blck;

    // 逐行量化权重
    size_t row_size = ggml_row_size(type, n_cols);
    std::vector<uint8_t> wq((size_t)n_rows * row_size);
    std::vector<float> row(n_cols);
    for (int64_t r = 0; r < n_rows; r++) {
        for (int64_t i = 0; i < n_cols; i++) row[i] = (float)((rand()%2000)-1000)/500.0f;
        auto rb = quantize_row(type, n_cols, row.data());
        memcpy(wq.data() + (size_t)r*row_size, rb.data(), row_size);
    }
    // 输入
    std::vector<float> in(n_cols * batch, 0.5f);

    struct ggml_init_params p = { .mem_size = 1024*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(p);

    // a: [ne0=n_cols, ne1=n_rows] 量化； b: [ne0=n_cols, ne1=batch] f32
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, n_cols, n_rows);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, batch);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Allocate all tensors using backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, B.backend);
    if (buf == NULL) {
        fprintf(stderr, "failed to allocate tensors for mul_mat\n");
        ggml_free(ctx);
        return;
    }

    // Set input data
    ggml_backend_tensor_set(a, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(b, in.data(), 0, in.size()*sizeof(float));

    // warmup
    ggml_backend_graph_compute(B.backend, gf);

    // timed
    double t0 = now_s();
    for (int i = 0; i < B.iters; i++) {
        ggml_backend_graph_compute(B.backend, gf);
    }
    double sec = (now_s() - t0) / B.iters;

    double work = (batch == 1) ? (double)n_rows : (double)n_rows * batch;
    const char * tag = (batch == 1) ? "mmvq" : "mmq";
    printf("%-22s %12.6f %14.3f M %s/s\n",
           (std::string(ggml_type_name(type))+" "+tag).c_str(), sec, work/sec/1e6, tag);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    int64_t n_rows  = argc > 1 ? atoll(argv[1]) : 4096;
    int64_t n_cols  = argc > 2 ? atoll(argv[2]) : 4096;
    int64_t mmq_batch = argc > 3 ? atoll(argv[3]) : 64;
    int iters    = argc > 4 ? atoi(argv[4]) : 50;
    srand(1234);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "cuda backend init failed\n"); return 1; }

    // 列对齐到 256（GSQ2=128 / Q2_K=256 的公倍数）
    n_cols = (n_cols / 256) * 256;
    if (n_cols < 256) n_cols = 256;

    printf("=== Kernel micro-benchmark: GSQ2 vs Q2_K ===\n");
    printf("n_rows=%lld n_cols=%lld mmq_batch=%lld iters=%d\n\n",
           (long long)n_rows, (long long)n_cols, (long long)mmq_batch, iters);

    Bench B{backend, iters};

    printf("%-22s %12s %16s\n", "kernel", "sec/iter", "throughput");
    printf("----------------------------------------------------------\n");

    printf("\n[DEQUANT] n=%lld\n", (long long)(n_rows*n_cols));
    bench_dequant(B, GGML_TYPE_GSQ2, n_rows*n_cols);
    bench_dequant(B, GGML_TYPE_Q2_K, n_rows*n_cols);

    printf("\n[MMVQ] batch=1 (mul_mat_vec_q)\n");
    bench_mul_mat(B, GGML_TYPE_GSQ2, n_rows, n_cols, 1);
    bench_mul_mat(B, GGML_TYPE_Q2_K, n_rows, n_cols, 1);

    printf("\n[MMQ] batch=%lld (mul_mat_q)\n", (long long)mmq_batch);
    bench_mul_mat(B, GGML_TYPE_GSQ2, n_rows, n_cols, mmq_batch);
    bench_mul_mat(B, GGML_TYPE_Q2_K, n_rows, n_cols, mmq_batch);

    printf("\n注:\n  dequant: M 元素/秒\n  mmvq: M 行/秒 (batch=1=单token)\n  mmq: M 点积/秒 (行×batch)\n");
    ggml_backend_free(backend);
    return 0;
}
