### A. 问题分析与方案总述
- 问题分析：
  1. 顺序 MTP：每个 draft token 一次 `llama_decode` + host top-k。75k + FlashAttn 下 draft 图启动/调度占明显份额；`n_max=2` 已是吞吐甜点，再加 N 几乎只掉 accept、不涨 `mean len`。
  2. 多步 MTP 的 token/hidden 依赖是确定的（greedy），不必回到 host 再 enqueue。
  3. CLI 若做成纯开关，`is_truthy("1")` 会吞掉「深度=1」；必须区分 truthy / falsey / 数字 N。
- 方案总体思路：opt-in `--spec-chain`。投机层组「catch-up 行 + N 个 chain 行」一次 `llama_decode`，`llama_set_mtp_chain(true)` 让 DECODER_MTP 在图内 argmax→embed→下一步。比选：保持顺序 MTP 为默认（正确性/多序列/backend sampling 更简单）。chain 只服务「独立 draft ctx、单 MTP head、单序列 drafting」。verify 完全不动。

### B. 代码定位
- `common/common.h` — `common_params_speculative_draft::chain`（默认 false）
- `common/arg.cpp` — `--spec-chain` / `LLAMA_ARG_SPEC_CHAIN`：truthy→开；falsey→关；整数 N≥1→开且 `n_max=N`
- `common/speculative.cpp` — `chain_enabled`；`chain_graph`；chain 时禁用 backend sampler；`generate` 单序列分支：`llama_set_mtp_chain` + 读 `[id,p]` 对
- `src/llama-cparams.h` / `src/llama-context.h` / `src/llama-ext.h` — `mtp_chain`、`set_mtp_chain`、`llama_set_mtp_chain`
- `src/llama-graph.h` — graph cache 比较含 `mtp_chain`（避免与普通 MTP 图混用）
- `src/llama-context.cpp` — 默认 false；logits 抽取按 `ggml_nbytes`（`process_ubatch` 与另一处 extract 两处）
- `src/models/qwen35.cpp` / `src/models/qwen35moe.cpp` — DECODER_MTP unroll：catch-up 只写 KV；chain 行图内 head / `LLAMA_SPEC_CHAIN_SUB` / argmax+softmax 写出 `[2,n_chain]`
- `tools/server/server-context.cpp` — MTP ctx `n_outputs_max >= n_parallel * n_max`

### C. 调用链影响
- `llama_set_mtp_chain` — 调用方：`common/speculative.cpp` chain 分支（decode 前后开关）
- `cparams.mtp_chain` — 消费方：两套 `build_qwen35*` DECODER_MTP；`llama-context` logits 拷贝；`llm_graph_params` 相等性
- `--spec-chain` — `common_params_parser`；server 读 `params.speculative.draft.chain` 与 `LLAMA_SPEC_CHAIN`
- 默认 MTP `generate` 顺序路径 — chain 关闭时行为不变（backend sampling 仍可用）

### E. 测试要点
- [unit / compile] `llama-server`（及依赖的 `common` / `llama`）增量编译通过
- [unit / CLI] `test-arg-parser` 现有用例不回退；`ggml.ai` 非 200 时跳过 download 段（避免 infra abort）
- [unit / 静态] `--spec-chain` 解析（`test-arg-parser`）：`1` 只置 `chain` 且不改默认 `n_max`；`2` 同时置 `n_max=2`；`off` 关闭
- [unit / 静态] `mtp_chain` 进入 `llm_graph_params` 比较（无独立单测，靠代码审查）
- [integration / 人工] 75k `-cmoe` greedy 500 tok：`--spec-chain 2` + `LLAMA_SPEC_CHAIN_SUB=0` + SWA，吞吐不低于同条件顺序 `n_max=2`；中文可读
- [integration / 对齐] 8k greedy：`--spec-chain 2` 与顺序 MTP `n_max=2` token 全等；二者与关 MTP 在第 13 token 分叉（既有 MTP 差异，非 chain 回归）
- [integration / 回归] 默认不传 `--spec-chain` 时行为与本需求前 MTP 一致
- [integration / 陷阱] 单独 `--spec-chain 1` 实际深度为默认 `n_max`（3），文档/bench 禁止这样开

### G. 风险待办
- `LLAMA_SPEC_CHAIN_SUB` 默认 32768：中文 draft 几乎全废（accept ~0.22、乱码）。触发：未 export `=0`。影响：误判 chain「更慢/更差」。缓解：bench 脚本与 clarification §B 写死。
- `--spec-chain 1` CLI 陷阱：深度不是 1。影响：和 `n_max=2` 对比失真。
- 多序列 / shared-mem / `chain_heads`：不走 `chain_graph`，静默回退顺序路径。影响：以为开了 chain 实际没有。
- 工作区 `testing/CMakeLists.txt` 是无关 CUDA bench 改动，提交必须排除。

### Env 执行环境提示
- 本次改动能否在编码环境内做语法/类型级自检？**能**（本机已编过 `llama-server` 并跑 75k bench；本回合不重编）
- 是否涉及需要容器内才能验证的行为？**是**（MTP + CUDA + GSQ2 GGUF；图级正确性本机集成，无容器 GPU 则 integration 仍须人工）
- 建议的验证范围（增量单测目标）：`test-arg-parser`（回归）；权威验收是 `llama-server` compile + 75k 人工 bench
- 是否预判需要集成测试：**是**；自动化程度：**需人工触发 + 人工判定**
