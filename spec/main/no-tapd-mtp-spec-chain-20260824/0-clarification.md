---
spec_version: "1"
spec_type: change
id: "no-tapd-mtp-spec-chain-20260824"
status: draft
subdomain: main
cross_subdomains: []
complexity: L
work_type: perf
---

# MTP spec-chain：一次 decode 展开 N 步 draft

按「投机草稿路径 + 上下文 API + 两套 NextN 构图 + server 输出预算 / 单仓单人」定为 **L**。

本目录为**事后补齐**（代码已在工作区，会话口述要求按 `dev-workflow` 落 spec 并提交）。`/ai-verify` revision 1：compile pass；`test-arg-parser` 在 `http://ggml.ai/` 下载断言失败（`test_infra_issue`），不编造 unit/integration 通过。

### A. 工作类型（work_type）【必填，先于目标】
`perf` — 性能优化（吞吐/时延/资源，默认不改对外语义）

次要动机：给 `--spec-type draft-mtp` 增加可选构图（in-graph argmax 串联），减少 draft 侧 `llama_decode` 次数；主类型仍为 perf。verify 路径与 target 采样不变。

### B. 目标（goal）
优化对象是 MTP draft：现行路径对每个 draft token 做一次独立 `llama_decode` + host top-k；`--spec-chain N` 改为一次 decode 在图内展开 N 步（上一步 argmax + hidden 喂下一步）。范围：`common/speculative.cpp`、`llama_set_mtp_chain` / `cparams.mtp_chain`、Qwen3.5 / Qwen3.5-MoE `DECODER_MTP` 构图、server MTP `n_outputs_max`、CLI `--spec-chain`。不改 GSQ2 内核、不改 target verify。默认关闭。基线（本机 75k、-cmoe、`max_tokens=500`、SWA 开启）：顺序 MTP `n_max=2` ~55 t/s；配对实测 chain=2 相对同 N 约 +2%（峰值约 56.2 t/s）。**输出语义：accept 路径与顺序 MTP 相同；chain 只改 draft 构图与采样落点（图内 greedy argmax）。中文词表须 `LLAMA_SPEC_CHAIN_SUB=0`，否则默认截到 32768。勿单独传 `--spec-chain 1`（`is_truthy` 只开开关、不改 `n_max`，默认深度 3）。**
