## 0. Summary Card【必读】
- 检索是否命中：是；命中文档数：3（另有 1 篇低相关度未列出：testing-quantize-fns-thresholds.md，本轮不改阈值）
- 关键结论数：5；待确认缺口数：1（缺口 1 已由 design 闭合并修正）
- 最近一次更新阶段：design

## A. 检索到的相关文档【clarify 起草，design 可追加】
- `ai_docs/gsq2-format-contract.md` — 相关原因：本需求的权重侧格式契约 — 结论摘要：LSB-first bit 序、`code-2` 码本、128 元素块跨 4 个 q8_0、改核必须过 `test-gsq2-vec-dot`；本需求**不改权重侧**，契约继续适用。
- `ai_docs/testing-quantized-vec-dot.md` — 相关原因：现有 `test-gsq2-vec-dot` 的对照设计依据 — 结论摘要：对照侧量化必须走 CPU backend `ggml_cpy`（同 mul_mat `from_float`），禁 `from_float_ref`；本需求沿用现有测试即自动满足，新增 mul_mat_id 用例同样遵守。
- `ai_docs/verify-clean-tree-isolation.md` — 相关原因：本需求与上轮改动同文件，verify 隔离规范直接适用 — 结论摘要：verify 须 stash + 干净树编译 + 符号探针；本轮工作区还叠加了 q3_K CUDA 移植（另一独立任务），隔离时一并 stash。
- [design 补充] `ggml/src/ggml-cpu/ggml-cpu.c:233-238` — 相关原因：GSQ2 CPU 分发点 — 结论摘要：`.vec_dot = ggml_vec_dot_gsq2_q8_0`、`.vec_dot_type = Q8_0`；本需求改 vec_dot 语义契约但不动 traits 表项本身。
- [design 补充] `ggml/src/ggml-cpu/ops.cpp:8440-8446` — 相关原因：CPU fattn 也经 traits 消费 vec_dot（§C 缺口 1 闭合） — 结论摘要：`kq_vec_dot = type_traits_cpu(k->type)->vec_dot`，Q 按 `vec_dot_type` 的 `from_float` 以**标准 q8_0 布局**量化——GSQ2 K 下喂 wide 内核会静默错算，须显式 assert 拒绝（本需求在 §B 落地）。

## B. 关键结论（供 go 编码遵守）【必填，可为空但需说明为空原因】
- 权重侧 `block_gsq2` 布局、量化/反量化、GGUF、CUDA 一律不动（gsq2-format-contract 契约）；wide 只改**激活侧**布局与内核消费方式。
- 激活置换量化的字节必须与标准 Q8_0 路径逐位一致（`ggml_quantize_row_gsq2_act` 内部调派发的 `quantize_row_q8_0`），`test-gsq2-vec-dot` 不修改即应全绿。
- `block_gsq2_act`（192B/128 元素）只存在于 `params->wdata`，mul_mat / mul_mat_id / graph_plan 三处行大小与预留必须联动，llamafile sgemm 路径须跳过 GSQ2。
- [design 补充] CPU fattn（`ops.cpp`）必须显式拒绝 GSQ2 K/V（`GGML_ASSERT`）——这是第三条 vec_dot 消费路径，防静默错算的唯一屏障。
- [design 补充] 非 AVX2 generic 回退与 AVX2 内核必须消费**同一** `block_gsq2_act` 布局（单一激活布局，防两套语义并存）。

## C. 待确认缺口（ai_docs 未覆盖，凭经验假设的点）【必填，可为空】
- ~~GSQ2 CPU 消费路径仅 mul_mat/mul_mat_id~~ — **已闭合（design）**：反搜证明假设不成立，CPU fattn（`ops.cpp:8440`）是第三条路径；处置=fattn 入口 assert 拒绝（见 1-change-plan §B）。若 `->vec_dot` 反搜仍有遗漏路径，影响同前（静默错算）；verify 阶段复核检索即可。
- 求和顺序重排的精度可接受性 — 假设：可接受（预研 200 组随机对照 mean rel err 3.4e-07 vs 现行 2.3e-07，均为 fp32 舍入级）— 影响：若验收要求位级一致需回退方案；已在 change-plan §E 落"精度"测试点（现有 1e-4/1e-5 容差覆盖）。

## D. 供 review 核对的约束清单【必填，可为空】
- Diff 限于六个文件：`ggml/src/ggml-cpu/quants.c`、`quants.h`、`arch/x86/quants.c`、`ggml-cpu.c`、`ops.cpp`（仅 fattn guard）、`tests/test-gsq2-vec-dot.cpp`（仅追加 mul_mat_id 用例）。[design 修订：较 clarify 初稿的 4 文件增加 ops.cpp guard 与测试扩展]
- 不改 `block_gsq2` 布局 / 量化公式 / GGUF / CUDA / `ggml_get_type_traits` 对外语义 / traits 表项。
- 非 AVX2 的 generic 回退必须消费同一 `block_gsq2_act` 布局（不允许两套激活布局并存）。
- GSQ2 权重 + 预量化 Q8_0 src1 直通路径显式 `GGML_ASSERT(F32)`；CPU fattn 显式拒绝 GSQ2 K/V（两处行为变化均在 coding-snapshot 记录理由）。
- 现有 `test-gsq2-vec-dot.cpp` 既有 5 用例不修改即应全绿；mul_mat_id（MoE 路径）必须有对照测试。
- `testing/CMakeLists.txt` 不在本需求范围（q3_K 任务的改动不得混入本需求提交）。
