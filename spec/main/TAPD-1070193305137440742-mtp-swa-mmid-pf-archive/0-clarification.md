---
spec_version: "1"
spec_type: change
id: "1070193305137440742"
status: draft
subdomain: main
cross_subdomains: []
complexity: M
work_type: perf
---

# MTP 草稿滑动窗口 + MoE 专家预取（补 spec 归档）

> TAPD 单 [#1070193305137440742](https://tapd.woa.com/tapd_fe/70193305/story/detail/1070193305137440742) 拉取失败（重定向到企业登录页），用户手动贴出单据正文作为澄清素材，`source.type` 仍记为 `tapd`，`fetch_status=degraded`，事后可核对。

按「单仓单人；两个已完成 commit（`llama-model.cpp`/`server-context.cpp`/`ggml-cpu.c` 共 3 个文件）仅归档不改代码，本次新增改动（`n_ubatch<=W` 运行时检查 + 2 个单测）预计另涉及 1~2 个文件」定为 **M**。

### A. 工作类型（work_type）【必填，先于目标】
`perf` — 性能优化（吞吐/显存，默认不改对外语义）

### B. 目标（goal）

TAPD 单描述的两个优化点**代码均已实现并提交**（`git log` 可查，工作区无未提交改动），本次诉求是**补 spec 归档**：以已有实现为准写清目标与验收标准，`design` 阶段从简，重心放在补单测 + verify + review，让这两个提交进入规范流程存档；同时顺带补一个已发现的运行时保护缺口。

1. **MTP 草稿滑动窗口（`LLAMA_MTP_SWA`，已提交 `1b32c65da`）**：MTP 自推测解码在 8GB 显卡 `-c 393216` 下因草稿上下文继承目标模型完整 per-slot 长度而不可用（单层 nextn 分配 768 MiB f16 KV + 等量 FlashAttention f16 反量化 scratch，显存冲到 ~7.4 GB；量化草稿 KV / offload 到主存均无效，CUDA FA MMA scratch 按元素数而非位宽计算）。设置 `LLAMA_MTP_SWA=W` 后：`llama-model.cpp` 为 `LLAMA_CONTEXT_TYPE_MTP` 的 KV cache 接入 `swa_type=STANDARD`/`n_swa=W`（旧 cell 被 mask 并淘汰）；`server-context.cpp` 同步把草稿上下文按 `2*W*n_parallel` 封顶（窗口 + 一个在途 ubatch，**要求 `n_ubatch<=W`**）。实测（`gsq2-nonexpert-q6k-v3-mtp-q4k` @ 75k context，`-np 3`，`--spec-draft-n-max 2`，`W=4096`）：草稿 KV 768→48 MiB，设备显存 7.4→6.5 GB，decode ~42.6 t/s（该 context 下不开启则功能直接不可用）。目标模型仍逐 token 验证草稿输出（lossless speculation），SWA 只影响接受率（速度），不改变输出；不设置该变量时行为与上游完全一致（no-op）。

2. **MoE 专家权重行软件预取（`GGML_MMID_PF`，已提交 `00f897882`）**：`--cpu-moe` 时 CPU `mul_mat_id` 内循环每步从 DRAM 流式读取多个专家权重矩阵（~30 GB/s 持续读取），双跑探针测得 ~20% 的算子耗时是暴露的 DRAM 延迟而非带宽。在内循环中提前 `pf_distance`（默认 8，实测扫参 4/6/8/12/16，峰值在 8）行预取即将点积的下一专家权重行，缓解 chunk 头部与 expert 切换处的延迟残留。实测（`gsq2-nonexpert-q6k-v3-mtp-q4k`，RTX 5060 8GB / Ultra 7 265KF，`-t 8 -tb 8 -cmoe -ngl 999`，MTP `draft-mtp n_max=2`，greedy）：75k context `48.73→50.06 t/s`（+2.7%），8k context `53.56→55.89 t/s`（+4.4%）；prefill/TTFT/显存不变。纯预取 hint，**数值逐位一致**；`GGML_MMID_PF=0` 关闭。

3. **本次归档新增（加固已知缺口，非新特性）**：`server-context.cpp` 注释写明 `Requires n_ubatch<=W`，但当前**无运行时强制检查**——`n_ubatch>W` 时草稿滑动窗口容量与在途 ubatch 的覆盖关系被破坏，存在静默越界风险。本次在现有代码基线上补一个运行时 assert/报错，拒绝该非法组合（不影响未设置 `LLAMA_MTP_SWA` 时的默认路径）；并分别为 `LLAMA_MTP_SWA`（KV cache 尺寸/`swa_type` 切换）与 `GGML_MMID_PF`（预取开关的数值一致性）新增单测用例（命名参照现有 `tests/test-gsq2-vec-dot.cpp` 风格）。

**范围边界**：两个开关默认关闭时行为不变（均为 no-op）；不改量化格式/输出数值语义；本次代码改动仅限 `n_ubatch<=W` 运行时检查 + 新增单测，不重新设计已提交的 SWA/预取实现本身。**归档 + 补一处运行时保护 + 补两组单测，不引入新的对外行为。**
