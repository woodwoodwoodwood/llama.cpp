---
spec_version: "1"
spec_type: change
id: "no-tapd-gsq2-cpu-avx2-vnni-20260818"
status: ready
subdomain: main
cross_subdomains: []
complexity: M
work_type: perf
---

# GSQ2 CPU AVX2/AVX_VNNI 点积核

按「核 2 文件 + 对照单测 / 单仓单人」定为 **M**。

### A. 工作类型（work_type）
`perf` — 性能优化（吞吐/时延/资源，默认不改对外语义）

次要动机：补齐 x86 上 GSQ2 相对 IQ2_M / Q4_K 等已有向量核的实现缺口；主类型仍为 perf，不升为 feature。

### B. 目标（goal）
为 GSQ2 在 x86 上补 AVX2 + AVX_VNNI 的 `ggml_vec_dot_gsq2_q8_0`，替换标量 generic，并在同一提交里加上对照单测。范围是 CPU 点积核、`arch-fallback.h` 的 x86 别名、以及 `test-gsq2-vec-dot`；不改量化格式、不改 CUDA。基线：`-cmoe` decode 约 16 t/s；目标：约 40 t/s（对齐同机 IQ2_M 带宽上限）。分支 `gsq2-quant-type`。

**默认不改对外数值语义，只把 GSQ2 CPU decode 从标量核拉到与同机其它低比特核同一带宽档。**