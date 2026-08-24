## 0. Summary Card【必读】
- 检索是否命中：是；命中文档数：1（另有 3 篇低相关度未列出：gsq2-format-contract / testing-quantized-vec-dot / testing-quantize-fns-thresholds，本需求不改量化）
- 关键结论数：4；待确认缺口数：2
- 最近一次更新阶段：design
- `ai_docs/modules.md`：不存在 → 子域默认 `main`

## A. 检索到的相关文档【clarify 起草，design 可追加】
- `ai_docs/verify-clean-tree-isolation.md` — 相关原因：工作区同时有无关 diff，verify 必须隔离 — 结论摘要：stash 非本需求改动后再编；本需求**不得**混入 `testing/CMakeLists.txt`（bench-kernels CUDA 规则，另一任务）。
- [design 补充] `common/speculative.cpp` MTP ctor — `chain_graph = !is_mem_shared && !chain_heads && chain_enabled`；`LLAMA_SPEC_CHAIN` 非空即开（覆盖 `--no-spec-chain`）。
- [design 补充] `src/models/qwen35moe.cpp` / `qwen35.cpp` DECODER_MTP — `mtp_chain && n_tokens>1 && n_seqs_unq==1` 才走 unroll；logits 变为 `[2, n_chain]`（id, p）。

## B. 关键结论（供 go 编码遵守）
- chain 默认关；verify / target 采样 / GSQ2 内核一律不动。
- 图内采样是 greedy argmax；`LLAMA_SPEC_CHAIN_SUB` 未设 = 32768（Qwen 中文 id 会被截掉），中文必须 `0`。
- `--spec-chain`：truthy 只开 `chain`；数字 N≥1 同时设 `n_max=N`。深度 1 应写 `--spec-chain 2 --spec-draft-n-max 1` 或只用 `--spec-draft-n-max 1`。
- 仅单序列 drafting 走 chain batch；多序列回退顺序 decode。chain 时关掉 backend sampler offload。

## C. 待确认缺口（ai_docs 未覆盖，凭经验假设的点）
- 无 dedicated spec-chain 单测 — 假设：本轮不补图级单测，验收靠 75k 集成 + 可读抽检 — 影响：容器 verify 只能 compile + 现有 `test-arg-parser`（未覆盖新 flag）。
- `qwen35.cpp` 与 `qwen35moe.cpp` 两份 unroll 必须保持同契约 — 假设：已镜像实现 — 影响：只改一侧会在非 MoE Qwen3.5 MTP 上静默分叉。

## D. 供 review 核对的约束清单
- Diff 限于：`common/{arg.cpp,common.h,speculative.cpp}`、`src/llama-{context.cpp,context.h,cparams.h,ext.h,graph.h}`、`src/models/{qwen35.cpp,qwen35moe.cpp}`、`tools/server/server-context.cpp`，以及本 spec 目录。
- 不提交 `testing/CMakeLists.txt`、`.cursor/`、其它 spec 快照、`ai_docs/`。
- `n_outputs_max` 在 chain 开启时至少 `n_parallel * n_max`。
- logits 抽取在 `mtp_chain` 下按 `ggml_nbytes(t_logits)`，不得按 `n_tokens * n_vocab`。
