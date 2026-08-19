---
spec_version: "1"
spec_type: change-plan
id: "no-tapd-gsq2-cpu-avx2-vnni-20260818"
status: draft
subdomain: main
work_type: perf
complexity: M
---

# Change plan：GSQ2 CPU AVX2/AVX_VNNI 点积核

一次提交同时落地 x86 点积核与对照单测。分支 `gsq2-quant-type`。不改量化格式、不改 CUDA。

### A. 问题分析与方案总述
- 问题分析：`type_traits_cpu[GGML_TYPE_GSQ2].vec_dot` 指向 `ggml_vec_dot_gsq2_q8_0`（`ggml/src/ggml-cpu/ggml-cpu.c:235`）。改核前 x86 的 `arch-fallback.h` 把 `ggml_vec_dot_gsq2_q8_0_generic` alias 成同一符号，于是 `arch/x86/quants.c` 没有独立实现，运行的是 `ggml/src/ggml-cpu/quants.c:180` 的标量循环（每字节拆 4 个 2-bit，再 `code-2` 乘 Q8）。同文件里 Q4/IQ2 等已有 AVX2 + `mul_sum_i8_pairs_float`。`-cmoe` 时专家权重走这条 CPU `vec_dot`，标量核把 decode 卡在约 16 t/s。仓库也没有 GSQ2×Q8 对照单测；`test-quantize-fns.cpp` 虽会扫到 GSQ2，却被 `if (NOT GGML_BACKEND_DL)` 挡住。bit 序或 `code-2` 错会静默错算。CUDA 已有 `vec_dot_gsq2_q8_1`，不在本次范围。
- 方案总体思路：按现有 x86 低比特核的模式补 `ggml_vec_dot_gsq2_q8_0`（AVX2 unpack → `code-2` → `mul_sum_i8_pairs_float`），并去掉仅 x86 的 generic alias。同一提交里加 `test-gsq2-vec-dot`：经 CPU `mul_mat` 打到优化核，对照测试内复刻的 generic，锁 LSB-first 与数值。选复用 `mul_sum_i8_pairs_float` 而不是另写累加，是为了和 Q4_0/IQ 路径同一套 VNNI/maddubs 语义。

### B. 代码定位
- `ggml/src/ggml-cpu/arch/x86/quants.c` — 增加 `bytes_from_2bit_32`（约 L98–113）与 `ggml_vec_dot_gsq2_q8_0`（约 L720–761）。签名：`void ggml_vec_dot_gsq2_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc)`。`#if defined(__AVX2__)`：每个 GSQ2 block（`QK_GSQ2=128`）拆 4 次、每次 32 元，对应 4 个 `block_q8_0`；`#else` 调 `ggml_vec_dot_gsq2_q8_0_generic`。
- `ggml/src/ggml-cpu/arch-fallback.h` — 仅在 x86 `#elif`（约 L85–111）删除 `#define ggml_vec_dot_gsq2_q8_0_generic ggml_vec_dot_gsq2_q8_0`。generic / ARM / POWERPC / 其它分支的 alias 保留。
- `tests/test-gsq2-vec-dot.cpp` — 新增。`0xE4` 打包经 `to_float` 得到 `{-2,-1,0,1}` 循环；CPU `mul_mat(GSQ2, F32)` 与本地 `vec_dot_gsq2_q8_0_ref` 在 atol=1e-4 / rtol=1e-5 内一致。形状：`(1,1,128)` / `(8,1,128)` / `(4,3,256)`，外加固定 `0xE4` 一块。
- `tests/CMakeLists.txt` — 在 `test-backend-ops` 旁、`NOT GGML_BACKEND_DL` 块外注册 `llama_build_and_test(test-gsq2-vec-dot.cpp)`，`GGML_BACKEND_DL=ON` 也能编。
- `tests/test-quantize-fns.cpp` — GSQ2 量化/点积误差走独立阈值（`d=amax/2` + `{-2,-1,0,1}` 对 `0.1+2*cos` 过不了 generic 2-bit/lowbit）。仅非 DL 构建会编此 target。
- 不改：`block_gsq2`、quantize/dequant、`ggml-cpu.c` traits、CUDA、`testing/CMakeLists.txt`。

### C. 调用链影响
- `ggml_vec_dot_gsq2_q8_0` — 调用方：`ggml-cpu.c` type traits。上层 `ggml_mul_mat` / CPU backend 经 type traits 间接调用；单测也经 `mul_mat` 打到这条核。
- `ggml_vec_dot_gsq2_q8_0_generic` — 声明 `quants.h`；定义 `quants.c:180`。x86 AVX2 核的 `#else` 与非 x86 fallback 仍用它。单测不直链该符号（BACKEND_DL 下不可用），用测试内 ref 复刻其语义。
- `bytes_from_2bit_32` — 仅被本文件 `ggml_vec_dot_gsq2_q8_0` 使用。
- `ggml_get_type_traits(GGML_TYPE_GSQ2)->to_float` / `from_float_ref` — 单测用来锁 bit 序与打包；定义仍在 `ggml-quants.c`，不改。

### E. 测试要点
- [unit / compile] 编出 `llama-server` 与 `test-gsq2-vec-dot`；x86 变体（至少 haswell/alderlake）链接 `ggml_vec_dot_gsq2_q8_0`。通过：链接无 undefined reference。
- [unit / 静态] `arch-fallback.h` 的 x86 `#elif` 内不存在 GSQ2 generic→优化核 `#define`；其它 ISA 仍有。通过：检索二元判断。
- [unit / 数值] 跑 `test-gsq2-vec-dot`。通过：`0xE4` dequant 与所有 `mul_mat` 对照用例 exit 0。
- [unit / 缺口] `test-quantize-fns` 在 `GGML_BACKEND_DL=ON` 下仍不编；不伪称该 target 已跑。
- [integration / 人工] `marvis-local-bench` 75k `-cmoe`：decode 明显高于标量 ~16 t/s，进入约 40 t/s（对齐同机 IQ2_M）。
- [integration / 回归] 抽检一轮 500 token，文本可读。

### G. 风险待办
- unpack bit 序或 `code-2` 与 `quantize_row_gsq2_ref` / generic 不一致 → 全图 decode 静默错。缓解：unpack 与 generic L210–215 对齐；单测用 `0xE4` 锁 LSB-first。
- mul_mat 内部 Q8 量化若与 `from_float_ref` 不一致 → 单测假失败。缓解：Q8_0 的 from_float 与 ref 通常同一实现。
- 本机 `testing/CMakeLists.txt` 的 CUDA 补丁不在本次 diff。

### Env 执行环境提示
- 本次改动能否在编码环境内做语法/类型级自检？**能**。编 `llama-server` 与 `test-gsq2-vec-dot` 并跑单测。
- 是否涉及需要容器内才能验证的行为？**单测不依赖容器**；75k 性能仍依赖本机 GPU + GSQ2 GGUF。
- 建议的验证范围（增量单测目标）：`test-gsq2-vec-dot`。compile 含 `llama-server`。
- 是否预判需要集成测试：**是**（75k `-cmoe`）；自动化程度：**需人工触发 + 人工判定**。
