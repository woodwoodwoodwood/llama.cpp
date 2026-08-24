## 0. Summary Card【必读】
- 检索是否命中：否；命中文档数：0
- 关键结论数：3（clarify）+ 2（design 补充）= 5；待确认缺口数：1（1 条已由 design 查实关闭）
- 最近一次更新阶段：design

## A. 检索到的相关文档【clarify 起草，design 可追加】
（无。原因：`lijiajieli/llama.cpp` 仓库根目录下不存在 `ai_docs/` 目录，未检索到任何相关文档；未编造相关文档，见 F12。design 阶段同样未命中，不重复检索已确认为空的目录。）

## B. 关键结论（供 go 编码遵守）【必填，可为空但需说明为空原因】
- 为空。原因：ai_docs 无命中，本节不编造"通用最佳实践"充数。以下事实来自本次澄清/设计对代码/git 历史的直接调研（非 ai_docs 来源）：
  - `LLAMA_MTP_SWA` 逻辑分两处协同：`src/llama-model.cpp`（KV cache 的 `swa_type`/`n_swa`）+ `tools/server/server-context.cpp`（草稿上下文 `n_ctx` 按 `2*W*n_parallel` 封顶），两处改一处需同步核对另一处；本轮仅改 `server-context.cpp`。
  - `GGML_MMID_PF` 仅在 `ggml/src/ggml-cpu/ggml-cpu.c` 的 `ggml_compute_forward_mul_mat_id_one_chunk` 内循环生效，纯 CPU 路径，与 CUDA/其它 backend 无关。
  - `[design 补充]` `ggml_mmid_pf_distance()` 用 `static int d = -1` 做进程内一次性缓存，同一进程内无法先测一个 env 值再测另一个——对照测试必须通过 CTest 注册为不同进程（`ENVIRONMENT` 属性）。
  - `[design 补充]` `load_model` 唯一调用方是 `server.cpp:394`，失败统一按 `return 1`/"exiting due to model loading error" 处理，新增护栏走同一路径，调用方无需改动。

## C. 待确认缺口（ai_docs 未覆盖，凭经验假设的点）【必填，可为空】
- [x] `n_ubatch<=W` 运行时检查该放在哪一层 → 已确定放在 `server-context.cpp`（原地 inline，不抽取额外函数，用户明确要求）（由 change-plan 查实）
- TAPD 单原文本身只有两段技术描述，无独立验收标准/截止时间字段 — 假设：本文档 §B 的实测数据即为验收基准（不要求 design/verify 重新达到更高指标）— 影响：若产品侧另有验收线，需要事后补充到 TAPD。（design 阶段未进一步查实，保留待确认）

## D. 供 review 核对的约束清单【必填，可为空】
- `LLAMA_MTP_SWA`/`GGML_MMID_PF` 均未设置时行为与设置前完全一致（no-op），不得引入默认路径的行为变化。
- `GGML_MMID_PF` 是纯预取 hint，任何取值下计算结果必须与关闭时逐位一致（由两条 `GGML_MMID_PF` 取值下的 CTest 均需对照同一份 `vec_dot_gsq2_q8_0_ref` PASS 来验证）。
- `LLAMA_MTP_SWA` 只允许影响推测解码的接受率（速度），目标模型的最终输出不得改变。
- 新增的 `n_ubatch<=W` 运行时检查不得在未设置 `LLAMA_MTP_SWA`（`w==0`）时触发或产生额外开销。
- `LLAMA_MTP_SWA` 护栏为人工验证项，不要求自动化单测覆盖（用户明确认领）。
- `[design 补充]` `src/llama-model.cpp` 中独立的 `LLAMA_MTP_SWA` 解析逻辑本轮不改动，review 时不应误报"重复代码未消除"为本轮遗漏（已在 change-plan §G 记录为已知、非本轮范围）。
