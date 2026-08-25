## 0. Summary Card【必读】
- 检索是否命中：否；命中文档数：0（clarify）；代码调研已补
- 关键结论数：6；待确认缺口数：1
- 最近一次更新阶段：design --redo（补单测）

## A. 检索到的相关文档【clarify 起草，design 可追加】
（空）仓库无 `ai_docs/` 目录，也无 `ai_docs/modules.md`。clarify 未检索到相关文档，不编造（F12）。
- [design 补充] `ggml/src/ggml-cpu/ggml-cpu.c:233-238` — 相关原因：GSQ2 CPU 分发点 — 结论摘要：`.vec_dot = ggml_vec_dot_gsq2_q8_0`，`.vec_dot_type = Q8_0`。
- [design 补充] `ggml/src/ggml-cpu/quants.c:180-225` — 相关原因：标量真源 — 结论摘要：LSB-first 2-bit，`code-2`，一块 GSQ2 对 4 个 Q8_0。
- [design 补充] `ggml/src/ggml-cuda/vecdotq.cuh` `vec_dot_gsq2_q8_1` — 相关原因：排除 CUDA — 结论摘要：GPU 已有 GSQ2 点积，本需求不改。
- [design 补充] `tests/test-quantize-fns.cpp` + `tests/CMakeLists.txt:260` — 相关原因：既有量化/点积单测 — 结论摘要：会扫 GSQ2，但 `GGML_BACKEND_DL=ON` 不编此 target；本机预设编不进去。
- [design 补充] `tests/test-backend-ops.cpp` `all_types[]` — 相关原因：backend 对照 — 结论摘要：类型表无 `GGML_TYPE_GSQ2`，本轮不往该大套件里加（避免 CUDA 对照扩 scope）。

## B. 关键结论（供 go 编码遵守）【必填，可为空但需说明为空原因】
- x86 上 `ggml_vec_dot_gsq2_q8_0` 必须走 `arch/x86/quants.c` 的 AVX2 实现，禁止再把该符号 alias 到 generic。
- 2-bit 拆包须与 `ggml_vec_dot_gsq2_q8_0_generic` 一致：LSB-first，每字节 `c0..c3` 对应 bits `[1:0]..[7:6]`，再 `code-2` 得到 `{-2,-1,0,1}`。
- 点积走现有 `mul_sum_i8_pairs_float`（AVX_VNNI 的 `vpdpbusd`，无 VNNI 时 maddubs）；不要另写一套累加语义。
- [design 补充] 只删 `arch-fallback.h` **x86** 分支的 GSQ2 alias；其它 ISA 的 `#define` 必须保留。
- [design 补充] 不改 `block_gsq2`、quantize/dequant、`ggml-cpu.c` traits、CUDA；`testing/CMakeLists.txt` 不进本 MR。

## C. 待确认缺口（ai_docs 未覆盖，凭经验假设的点）【必填，可为空】
- 仓库无 GSQ2 CPU 核的书面约定 — 假设：数值必须与 generic 逐元素一致 — 影响：若 bit 序或 offset 与量化写入不一致，decode 会静默算错。
- [design 补充] 本机 `GGML_BACKEND_DL=ON` 不能直链 `ggml_vec_dot_gsq2_q8_0_generic` — 假设：CPU `mul_mat` + 测试内 ref 等价于 AVX2 vs generic — 影响：Q8 量化若与 ref 分叉会假失败。

## D. 供 review 核对的约束清单【必填，可为空】
- Diff 核部分限于 `ggml/src/ggml-cpu/arch/x86/quants.c` 与 `ggml/src/ggml-cpu/arch-fallback.h` 的 x86 分支（去掉 GSQ2 generic alias）。
- 不得改 `block_gsq2` 布局、量化/反量化公式、CUDA kernel、`ggml-cpu.c` traits。
- 非 AVX2 路径须回退 `ggml_vec_dot_gsq2_q8_0_generic`。
- [design 补充] 非 x86 的 fallback alias 不得一并删掉。
- [design 补充] 公开签名保持 `quants.h` 已有原型，无 API 变更。
- [design 补充] 本轮允许新增 `tests/test-gsq2-vec-dot.cpp` 与 `tests/CMakeLists.txt` 注册；`test-quantize-fns.cpp` 只加 GSQ2 阈值。
