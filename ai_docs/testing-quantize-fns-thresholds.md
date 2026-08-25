# test-quantize-fns 阈值分带规则（非对称 codebook 格式）

> 来源：spec/main/no-tapd-gsq2-cpu-avx2-vnni-20260818/（2026-08-19，经人工确认写入）

新增量化格式接入 `tests/test-quantize-fns.cpp` 时，若 codebook **非对称**（如 GSQ2：`d=amax/2`，值域 `[-amax, +amax/2]`，正侧只覆盖一半），对默认测试数据 `0.1 + 2*cos`（正峰在 +2.1）会出现大量正峰被夹 → 绝对 RMSE 贴通用 lowbit 阈值上沿、点积误差被夹位偏差主导，**过不了通用阈值不是内核错误**。

规则：
1. 此类格式须设**独立阈值常量**（如 `MAX_QUANTIZATION_TOTAL_ERROR_GSQ2`）并在常量处注释成因（非对称覆盖区间 + 实测值）
2. 阈值带参考：GSQ2 abs RMSE 实测 ~0.00796 → 限 0.010；dot 实测 ~0.213 → 限 0.25（x86 非 DL 构建证据）
3. **禁止改量化公式迁就测试**——公式变更破坏 GGUF 二进制兼容
4. 注意：该 target 仅非 `GGML_BACKEND_DL` 构建编译；DL 开发机上不跑是预期行为

来源证据：4-reviews/20260818-1805-go_only.md P1（FIX-1）；tests/test-quantize-fns.cpp:22-26 成因注释。
