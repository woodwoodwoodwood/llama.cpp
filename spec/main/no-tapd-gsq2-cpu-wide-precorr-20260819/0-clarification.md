---
spec_version: "1"
spec_type: change
id: "no-tapd-gsq2-cpu-wide-precorr-20260819"
status: draft
subdomain: main
cross_subdomains: []
complexity: M
work_type: perf
---

# GSQ2 CPU wide 解包 + 预计算修正内核

按「核 4 文件（quants.c/h、arch/x86/quants.c、ggml-cpu.c 调度接线）/ 单仓单人」定为 **M**。

### A. 工作类型（work_type）【必填，先于目标】
`perf` — 性能优化（吞吐/时延/资源，默认不改对外语义）

次要动机：替换上一轮（no-tapd-gsq2-cpu-avx2-vnni-20260818）交付的 narrow 解包内核，兑现其未达成的 ~40 t/s 目标值；主类型仍为 perf。

### B. 目标（goal）
优化对象是 GSQ2 CPU 点积路径（decode / MoE `-cmoe` 专家扫描）：现行 narrow 内核每 128 元素块花 ~60 条指令解包 2-bit code（`bytes_from_2bit_32` ×4），解包而非 dpbusd 是指令瓶颈（内核算力 ~40 GB/s，L3 常驻上限的 61%）。方案：wide 解包（整块 32B 一次 load，and/srli 4 向量 ~8 条指令）+ 激活侧置换成 `block_gsq2_act` 布局 + prep 预计算 `code-2` 偏移的 per-lane 修正（内循环 4 dpbusd + 1 fmadd/块）。范围边界：仅 CPU 侧（`quants.c/h`、`arch/x86/quants.c`、`ggml-cpu.c` 的 mul_mat/mul_mat_id 调度），不改量化格式/GGUF/CUDA；`block_gsq2_act` 只存在于 wdata，不落盘。基线/目标：75k `-cmoe` decode ~30 t/s → ~38 t/s（会话预研已实测 37.97 t/s、L3 内核算力 65.6 GB/s、`test-gsq2-vec-dot` 5/5 与 mul_mat_id 冒烟 PASS——预研数据供 design 参考，正式验证仍走 verify）。**数值语义：整数域结果与现行内核位级一致（int 域修正），仅浮点求和顺序重排（ulp 级）；权重侧格式与 bit 序完全不动。**
