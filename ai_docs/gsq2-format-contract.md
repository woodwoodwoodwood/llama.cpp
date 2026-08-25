# GSQ2 数值契约

> 来源：spec/main/no-tapd-gsq2-cpu-avx2-vnni-20260818/（2026-08-19，经人工确认写入）

CPU 上任何 ISA 实现的 `ggml_vec_dot_gsq2_q8_0` 必须与 `ggml_vec_dot_gsq2_q8_0_generic`（`ggml/src/ggml-cpu/quants.c`）**逐元素一致**：

1. **bit 序**：LSB-first，每字节 bits `[1:0]=c0, [3:2]=c1, [5:4]=c2, [7:6]=c3`（打包字节 0xE4 解出 codes {0,1,2,3} → 值 {-2,-1,0,1}）
2. **码本**：`w = (code - 2) * d`，code∈{0,1,2,3} → 有符号值 {-2,-1,0,1}；`block_gsq2` 布局（fp16 d + 32B qs）不得改动（GGUF 二进制兼容）
3. **块映射**：一个 128 元素 GSQ2 块跨 4 个 `block_q8_0`（32 元素/块），激活侧按 q8_0 逐子块带 scale
4. **验收**：改核必须通过 `tests/test-gsq2-vec-dot.cpp`（0xE4 bit 序锁定 + mul_mat 数值对照）

违反任何一条 → decode 静默错算（bit 序/offset 与量化写入不一致时不报错、只出错值）。

来源证据：0-context-snapshot.md §C 缺口 1（本轮证实）；4-reviews/20260819-1953-verify_informed.md §B。
