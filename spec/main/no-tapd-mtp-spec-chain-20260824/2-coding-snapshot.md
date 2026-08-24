## Summary Card【必读】
- 改动范围（implementation_scope）：`1-change-plan.md §B` 全部 11 个业务文件——CLI/`common_params`、投机 `chain_graph` 单序列路径、`llama_set_mtp_chain` + logits 按字节抽取、graph cache 键、`qwen35`/`qwen35moe` DECODER_MTP unroll、server `n_outputs_max`。另含本 spec 目录（事后补齐）。
- 调用链摘要（call_chain_summary）：`--spec-chain` / `LLAMA_SPEC_CHAIN` → `params.chain` → speculative ctor 置 `chain_graph` → 单序列 `generate` 组 catch-up+N 行 → `llama_set_mtp_chain(true)` → DECODER_MTP 图内 argmax 展开 → logits `[2,n_chain]` → host 只读 id/p。关 chain 后顺序 MTP + backend top-k 不变。
- 兼容风险（compat_risks）：默认关闭，旧命令行不变。行为变化仅 opt-in：图内 greedy（非 host top-k）；`LLAMA_SPEC_CHAIN_SUB` 默认截词表；`--spec-chain 1` 不解成深度 1。回滚：去掉 flag / 不设 env。
- 测试前置（test_prep）：VS2022+Clang+Ninja+CUDA 的 `build-x64-windows-llvm-release`；75k 集成需本机 GPU + GSQ2 GGUF + `marvis-local-bench`。
- 需回源（needs_reread）：`0-clarification.md §B`（SUB=0、禁止裸 `--spec-chain 1`）；`1-change-plan.md §E` / §G。

## Verification Handoff【必填】
- 本地自检能力：有编译条件，**本回合未重编**（会话前期已用该工作区二进制跑 75k；此字段不替代容器侧权威编译）。不得写成「本 commit 已在干净树上 compile/unit 通过」。
- 增量单测范围（unit_test_targets）：`test-arg-parser`
- 覆盖缺口对照（change-plan §E）：
  - [compile] `llama-server` → 会话前期编过，**本回合未重跑**；verify 必须自己编
  - [CLI] `test-arg-parser` 现有用例 + `--spec-chain` 三种形态 → **已落单测**；download 段在 `ggml.ai` 非 200 时跳过
  - [静态] graph params 含 `mtp_chain` → **已落码，无独立单测**
  - [integration] 75k chain=2 + SUB=0 → 提交态重跑 58.48 t/s / accept 0.733（近期同配方 54–59）；8k greedy 下 chain 与顺序 MTP token 全等
  - [integration] 默认关 chain 回归 → 未做独立 A/B（默认路径代码未改控制流入口）
  - [integration] `--spec-chain 1` 陷阱 → 已在文档标注，无自动化
- 是否需要集成测试：是；标注 `pending_integration_decision`
- 已知不确定项（供容器侧失败归因参考）：
  - **提交隔离**：只提交 §D 11 文件 + 本 spec。工作区另有 `testing/CMakeLists.txt`、`.cursor/`、旧 spec 快照——verify 若看到它们不得当成本需求。
  - `produced_by_commit` = `d554ecb12c9503b26121c42a465c9f4ec944ef5e`（实现 + 单测 + spec 折入后的 amend；live HEAD 可能仅 metadata 不同）。
  - 无图级单测：compile 通过不能证明 chain 数值/中文 accept。
  - Windows 上 `test-arg-parser` 跳过 env 段；`LLAMA_ARG_SPEC_CHAIN` 无单测。
