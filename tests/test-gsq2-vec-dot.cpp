// GSQ2 CPU vec_dot: AVX2/VNNI kernel vs scalar generic (same pack layout).
// Built even with GGML_BACKEND_DL=ON: goes through the CPU backend mul_mat path,
// and compares against a local copy of ggml_vec_dot_gsq2_q8_0_generic.

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)
#endif

static const char * RESULT_STR[] = { "ok", "FAILED" };

// Layout must match ggml-common.h: LSB-first 2-bit codes, 4 per byte.
constexpr int     QK_GSQ2_TEST = 128;
constexpr int     QK8_0_TEST   = 32;
constexpr uint8_t PACK_E4      = 0xE4; // bits 00,01,10,11 → codes {0,1,2,3}

struct block_gsq2_test {
    ggml_fp16_t d;
    uint8_t     qs[QK_GSQ2_TEST / 4];
};

struct block_q8_0_test {
    ggml_fp16_t d;
    int8_t      qs[QK8_0_TEST];
};

static_assert(sizeof(block_gsq2_test) == sizeof(ggml_fp16_t) + QK_GSQ2_TEST / 4,
              "GSQ2 test block must match ggml block_gsq2");
static_assert(sizeof(block_q8_0_test) == sizeof(ggml_fp16_t) + QK8_0_TEST,
              "Q8_0 test block must match ggml block_q8_0");

static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1f + 2.0f * cosf((float) i + offset);
    }
}

// Features of the *loaded* CPU backend DLL (compile-time ISA), not just host CPUID.
static bool cpu_backend_has_avx2(std::string & desc) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        desc = "no CPU backend";
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const char * name = ggml_backend_name(backend);
    const char * description = ggml_backend_dev_description(dev);
    bool has_avx2 = false;

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_features = (ggml_backend_get_features_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (get_features) {
        for (const ggml_backend_feature * f = get_features(reg); f && f->name; ++f) {
            if (std::strcmp(f->name, "AVX2") == 0 && f->value && std::strcmp(f->value, "1") == 0) {
                has_avx2 = true;
                break;
            }
        }
    }

    desc  = name ? name : "?";
    desc += " / ";
    desc += description ? description : "?";
    desc += has_avx2 ? " AVX2=1" : " AVX2=0";

    ggml_backend_free(backend);
    return has_avx2;
}

// Same Q8_0 path mul_mat uses (CPU from_float: x86 round-to-even), not from_float_ref.
static bool quantize_q8_0_cpu(const float * src, int64_t k, void * dst, size_t dst_bytes, std::string & err) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        err = "no CPU backend";
        return false;
    }

    struct ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        err = "ggml_init failed";
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * src_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    ggml_tensor * dst_t = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, k);
    ggml_tensor * cpy   = ggml_cpy(ctx, src_t, dst_t);
    ggml_cgraph * gf    = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cpy);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        err = "ggml_backend_alloc_ctx_tensors failed";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(src_t, src, 0, (size_t) k * sizeof(float));
    const enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        err = std::string("q8 cpy: ") + ggml_status_to_string(status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_get(dst_t, dst, 0, dst_bytes);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

// Same reduction as ggml_vec_dot_gsq2_q8_0_generic (quants.c).
static float vec_dot_gsq2_q8_0_ref(int n, const void * vx, const void * vy) {
    const int qk = QK_GSQ2_TEST;
    const int nb = n / qk;

    const block_gsq2_test * x = (const block_gsq2_test *) vx;
    const block_q8_0_test * y = (const block_q8_0_test *) vy;

    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d0 = ggml_fp16_to_fp32(x[i].d);
        float       sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            const block_q8_0_test * yb = &y[i * 4 + k];
            const float             d1 = ggml_fp16_to_fp32(yb->d);
            int                     sumi_block = 0;
            const uint8_t *         codes = &x[i].qs[k * 8];
            const int8_t *          qy    = yb->qs;
            for (int b = 0; b < 8; ++b, qy += 4) {
                const uint8_t byte = codes[b];
                sumi_block += (((int) ((byte) & 0x3) - 2) * qy[0])
                            + (((int) ((byte >> 2) & 0x3) - 2) * qy[1])
                            + (((int) ((byte >> 4) & 0x3) - 2) * qy[2])
                            + (((int) ((byte >> 6) & 0x3) - 2) * qy[3]);
            }
            sumi += d1 * (float) sumi_block;
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

static bool almost_equal(float a, float b, float atol, float rtol) {
    return fabsf(a - b) <= atol + rtol * fmaxf(1.0f, fabsf(b));
}

static int test_pack_e4_to_float(bool verbose) {
    const auto * qfns = ggml_get_type_traits(GGML_TYPE_GSQ2);
    if (!qfns || !qfns->to_float || qfns->blck_size != QK_GSQ2_TEST) {
        printf("pack 0xE4 to_float:                 FAILED (GSQ2 traits missing)\n");
        return 1;
    }
    if (qfns->type_size != sizeof(block_gsq2_test)) {
        printf("pack 0xE4 to_float:                 FAILED (type_size=%zu expected %zu)\n",
               qfns->type_size, sizeof(block_gsq2_test));
        return 1;
    }

    block_gsq2_test blk;
    blk.d = ggml_fp32_to_fp16(1.0f);
    memset(blk.qs, PACK_E4, sizeof(blk.qs));

    std::vector<float> out(QK_GSQ2_TEST, 0.0f);
    qfns->to_float(&blk, out.data(), QK_GSQ2_TEST);

    const float expect[4] = { -2.0f, -1.0f, 0.0f, 1.0f };
    int         failed    = 0;
    for (int i = 0; i < QK_GSQ2_TEST; i++) {
        if (out[i] != expect[i % 4]) {
            failed = 1;
            if (verbose) {
                printf("  idx %d: got %f expected %f\n", i, out[i], expect[i % 4]);
            }
            break;
        }
    }

    printf("pack 0xE4 to_float:                 %s\n", RESULT_STR[failed]);
    return failed;
}

static bool run_cpu_mul_mat_gsq2(
        int64_t m, int64_t n, int64_t k,
        const void * a_data, size_t a_bytes,
        const float * b_data, size_t b_bytes,
        float * out, size_t out_bytes,
        std::string & err) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        err = "no CPU backend (ggml_backend_load_all / GGML_BACKEND_DL)";
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        err = "ggml_init failed";
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_GSQ2, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        err = "ggml_backend_alloc_ctx_tensors failed";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(a, a_data, 0, a_bytes);
    ggml_backend_tensor_set(b, b_data, 0, b_bytes);

    const enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        err = std::string("ggml_backend_graph_compute: ") + ggml_status_to_string(status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_get(c, out, 0, out_bytes);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static int test_mul_mat_matches_generic(int64_t m, int64_t n, int64_t k, float offset, bool verbose) {
    const auto * gsq2 = ggml_get_type_traits(GGML_TYPE_GSQ2);
    if (!gsq2 || !gsq2->from_float_ref) {
        printf("mul_mat GSQ2 m=%lld n=%lld k=%lld:  FAILED (quantize ref missing)\n",
               (long long) m, (long long) n, (long long) k);
        return 1;
    }

    const size_t a_bytes = (size_t) m * ggml_row_size(GGML_TYPE_GSQ2, k);
    const size_t b_bytes = (size_t) n * ggml_row_size(GGML_TYPE_F32, k);
    const size_t q_bytes = (size_t) n * ggml_row_size(GGML_TYPE_Q8_0, k);

    std::vector<float>   a_f((size_t) m * (size_t) k);
    std::vector<float>   b_f((size_t) n * (size_t) k);
    std::vector<uint8_t> a_q(a_bytes);
    std::vector<uint8_t> b_q(q_bytes);
    std::vector<float>   got((size_t) m * (size_t) n, 0.0f);

    for (int64_t i = 0; i < m; i++) {
        generate_data(offset + (float) i, (size_t) k, a_f.data() + i * k);
        gsq2->from_float_ref(a_f.data() + i * k, a_q.data() + i * ggml_row_size(GGML_TYPE_GSQ2, k), k);
    }
    for (int64_t j = 0; j < n; j++) {
        generate_data(offset + 10.0f + (float) j, (size_t) k, b_f.data() + j * k);
        std::string qerr;
        if (!quantize_q8_0_cpu(b_f.data() + j * k, k, b_q.data() + j * ggml_row_size(GGML_TYPE_Q8_0, k),
                               ggml_row_size(GGML_TYPE_Q8_0, k), qerr)) {
            printf("mul_mat GSQ2 m=%lld n=%lld k=%lld:  FAILED (Q8 CPU quant: %s)\n",
                   (long long) m, (long long) n, (long long) k, qerr.c_str());
            return 1;
        }
    }

    std::string err;
    if (!run_cpu_mul_mat_gsq2(m, n, k, a_q.data(), a_bytes, b_f.data(), b_bytes,
                              got.data(), got.size() * sizeof(float), err)) {
        printf("mul_mat GSQ2 m=%lld n=%lld k=%lld:  FAILED (%s)\n",
               (long long) m, (long long) n, (long long) k, err.c_str());
        return 1;
    }

    int failed = 0;
    for (int64_t j = 0; j < n && !failed; j++) {
        const void * ycol = b_q.data() + j * ggml_row_size(GGML_TYPE_Q8_0, k);
        for (int64_t i = 0; i < m; i++) {
            const void * xrow = a_q.data() + i * ggml_row_size(GGML_TYPE_GSQ2, k);
            const float  ref  = vec_dot_gsq2_q8_0_ref((int) k, xrow, ycol);
            const float  out  = got[(size_t) i + (size_t) j * (size_t) m];
            if (!almost_equal(out, ref, 1e-4f, 1e-5f)) {
                failed = 1;
                printf("mul_mat GSQ2 m=%lld n=%lld k=%lld:  FAILED (i=%lld j=%lld got=%f ref=%f)\n",
                       (long long) m, (long long) n, (long long) k,
                       (long long) i, (long long) j, out, ref);
                break;
            }
        }
    }

    if (!failed && verbose) {
        printf("mul_mat GSQ2 m=%lld n=%lld k=%lld:  ok\n",
               (long long) m, (long long) n, (long long) k);
    } else if (!failed) {
        printf("mul_mat GSQ2 m=%lld n=%lld k=%lld:  ok\n",
               (long long) m, (long long) n, (long long) k);
    }
    return failed;
}

// mul_mat_id (MoE path) goes through the same wdata activation conversion in
// ggml-cpu.c, but a different chunk loop than plain mul_mat — exercise it with
// multiple experts and tokens. b -> [k, n_expert_used, n_tokens]; ids ->
// [n_expert_used, n_tokens]; c -> [rows, n_expert_used, n_tokens].
static bool run_cpu_mul_mat_id_gsq2(
        int64_t m, int64_t k, int n_as, int n_tok,
        const void * a_data, size_t a_bytes,
        const float * b_data, size_t b_bytes,
        const int32_t * ids_data, size_t ids_bytes,
        float * out, size_t out_bytes,
        std::string & err) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        err = "no CPU backend (ggml_backend_load_all / GGML_BACKEND_DL)";
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        err = "ggml_init failed";
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * a   = ggml_new_tensor_3d(ctx, GGML_TYPE_GSQ2, k, m, n_as);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_tok);
    ggml_tensor * c   = ggml_mul_mat_id(ctx, a, b, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        err = "ggml_backend_alloc_ctx_tensors failed";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(a, a_data, 0, a_bytes);
    ggml_backend_tensor_set(b, b_data, 0, b_bytes);
    ggml_backend_tensor_set(ids, ids_data, 0, ids_bytes);

    const enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        err = std::string("ggml_backend_graph_compute: ") + ggml_status_to_string(status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_get(c, out, 0, out_bytes);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static int test_mul_mat_id_matches_generic(int64_t m, int64_t k, int n_as, int n_tok, float offset, bool verbose) {
    const auto * gsq2 = ggml_get_type_traits(GGML_TYPE_GSQ2);
    if (!gsq2 || !gsq2->from_float_ref) {
        printf("mul_mat_id GSQ2 m=%lld k=%lld:       FAILED (quantize ref missing)\n",
               (long long) m, (long long) k);
        return 1;
    }

    const size_t a_bytes  = (size_t) n_as * m * ggml_row_size(GGML_TYPE_GSQ2, k);
    const size_t b_bytes  = (size_t) n_tok * ggml_row_size(GGML_TYPE_F32, k);
    const size_t q_bytes  = ggml_row_size(GGML_TYPE_Q8_0, k);
    const size_t out_bytes = (size_t) n_tok * m * sizeof(float);

    std::vector<float>   wf((size_t) n_as * m * k);
    std::vector<float>   xf((size_t) n_tok * k);
    std::vector<uint8_t> a_q(a_bytes);
    std::vector<uint8_t> b_q((size_t) n_tok * q_bytes);
    std::vector<int32_t> idv(n_tok);
    std::vector<float>   got(out_bytes / sizeof(float), 0.0f);

    for (int a = 0; a < n_as; a++) {
        for (int i = 0; i < m; i++) {
            generate_data(offset + a * 7.0f + (float) i, (size_t) k, wf.data() + ((size_t) a * m + i) * k);
            gsq2->from_float_ref(wf.data() + ((size_t) a * m + i) * k,
                                 a_q.data() + ((size_t) a * m + i) * ggml_row_size(GGML_TYPE_GSQ2, k), k);
        }
    }
    for (int t = 0; t < n_tok; t++) {
        generate_data(offset + 100.0f + (float) t, (size_t) k, xf.data() + (size_t) t * k);
        std::string qerr;
        if (!quantize_q8_0_cpu(xf.data() + (size_t) t * k, k,
                               b_q.data() + (size_t) t * q_bytes, q_bytes, qerr)) {
            printf("mul_mat_id GSQ2 m=%lld k=%lld:       FAILED (Q8 CPU quant: %s)\n",
                   (long long) m, (long long) k, qerr.c_str());
            return 1;
        }
    }
    for (int t = 0; t < n_tok; t++) {
        idv[t] = t % n_as; // round-robin expert routing
    }

    std::string err;
    if (!run_cpu_mul_mat_id_gsq2(m, k, n_as, n_tok,
                                 a_q.data(), a_bytes,
                                 xf.data(), b_bytes,
                                 idv.data(), idv.size() * sizeof(int32_t),
                                 got.data(), out_bytes, err)) {
        printf("mul_mat_id GSQ2 m=%lld k=%lld:       FAILED (%s)\n",
               (long long) m, (long long) k, err.c_str());
        return 1;
    }

    int failed = 0;
    for (int t = 0; t < n_tok && !failed; t++) {
        const int a = idv[t];
        for (int i = 0; i < m; i++) {
            const void * xrow = a_q.data() + ((size_t) a * m + i) * ggml_row_size(GGML_TYPE_GSQ2, k);
            const void * ycol = b_q.data() + (size_t) t * q_bytes;
            const float  ref  = vec_dot_gsq2_q8_0_ref((int) k, xrow, ycol);
            const float  out  = got[(size_t) t * m + i];
            if (!almost_equal(out, ref, 1e-4f, 1e-5f)) {
                failed = 1;
                printf("mul_mat_id GSQ2 m=%lld k=%lld:       FAILED (t=%d a=%d i=%lld got=%f ref=%f)\n",
                       (long long) m, (long long) k, t, a, (long long) i, out, ref);
                break;
            }
        }
    }

    printf("mul_mat_id GSQ2 m=%lld n_as=%d k=%lld:    %s\n",
           (long long) m, n_as, (long long) k, failed ? "FAILED" : "ok");
    return failed;
}

static int test_pack_e4_mul_mat(bool verbose) {
    const int64_t k = QK_GSQ2_TEST;
    const int64_t m = 1;
    const int64_t n = 1;

    block_gsq2_test blk;
    blk.d = ggml_fp32_to_fp16(0.5f);
    memset(blk.qs, PACK_E4, sizeof(blk.qs));

    std::vector<float> b_f((size_t) k);
    generate_data(3.0f, (size_t) k, b_f.data());

    std::vector<uint8_t> b_q(ggml_row_size(GGML_TYPE_Q8_0, k));
    std::string qerr;
    if (!quantize_q8_0_cpu(b_f.data(), k, b_q.data(), b_q.size(), qerr)) {
        printf("pack 0xE4 mul_mat:                  FAILED (Q8 CPU quant: %s)\n", qerr.c_str());
        return 1;
    }

    const float ref = vec_dot_gsq2_q8_0_ref((int) k, &blk, b_q.data());

    float       got = 0.0f;
    std::string err;
    if (!run_cpu_mul_mat_gsq2(m, n, k, &blk, sizeof(blk), b_f.data(), b_f.size() * sizeof(float),
                              &got, sizeof(got), err)) {
        printf("pack 0xE4 mul_mat:                  FAILED (%s)\n", err.c_str());
        return 1;
    }

    const int failed = !almost_equal(got, ref, 1e-4f, 1e-5f);
    if (failed || verbose) {
        printf("pack 0xE4 mul_mat:                  %s (got=%f ref=%f)\n", RESULT_STR[failed], got, ref);
    } else {
        printf("pack 0xE4 mul_mat:                  ok\n");
    }
    return failed;
}

int main(int argc, char ** argv) {
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    ggml_backend_load_all();

    std::string cpu_desc;
    const bool  has_avx2 = cpu_backend_has_avx2(cpu_desc);
    printf("CPU backend: %s\n", cpu_desc.c_str());

    int num_failed = 0;
    num_failed += test_pack_e4_to_float(verbose);
    if (!has_avx2) {
        printf("mul_mat GSQ2 AVX2 kernel tests:     SKIPPED (loaded backend has no AVX2; generic vs generic)\n");
    } else {
        num_failed += test_pack_e4_mul_mat(verbose);
        num_failed += test_mul_mat_matches_generic(1, 1, 128, 0.0f, verbose);
        num_failed += test_mul_mat_matches_generic(8, 1, 128, 1.0f, verbose);
        num_failed += test_mul_mat_matches_generic(4, 3, 256, 2.0f, verbose);
        num_failed += test_mul_mat_id_matches_generic(8, 256, 2, 3, 3.0f, verbose);
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }
    return num_failed > 0;
}
