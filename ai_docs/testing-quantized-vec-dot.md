# 量化点积单测对照规范（BACKEND_DL 构建）

> 来源：spec/main/no-tapd-gsq2-cpu-avx2-vnni-20260818/（2026-08-19，经人工确认写入）

写量化类型 × Q8 点积的对照单测时，**参考侧的激活量化必须经 CPU backend `ggml_cpy(F32→Q8_0)` 图计算**（即 mul_mat 内部分发的同一 `from_float`，x86 为 round-to-even 向量量化），**禁用 `from_float_ref`**。

理由：`from_float_ref` 与 CPU `from_float` 的量化字节可差 1 LSB；单元素差 1 LSB 时点积可偏约 `2*d0*d1`，远超常用 1e-4 容差 → **假失败**（测试报错但内核正确）。

参考实现：`tests/test-gsq2-vec-dot.cpp` 的 `quantize_q8_0_cpu()`（backend init → `ggml_cpy` 图 → tensor get）。

来源证据：4-reviews/20260818-1805-go_only.md P2（FIX-2）；tests/test-gsq2-vec-dot.cpp:81 注释。
