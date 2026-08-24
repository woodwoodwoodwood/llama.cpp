# no-tapd-mtp-spec-chain-20260824 — Review 报告

> Source: conversation  |  Subdomain: main  |  Complexity: L
> Date: 2026-08-24 18:00

## 0. Summary Card【必读】

- review_mode: verify_informed
- model: cursor-grok-4.6（⚠️ 协议偏离：`meta.yaml.models.review=opus`，本会话无法切换评审模型；沿本仓上一轮 wide-precorr 的显著标注继续。要严格合规请用 opus 复跑）
- 基于 3-verify-report.md revision 1（compile=pass, unit=fail / `test_infra_issue`, integration=not_run）
- Findings：P0=0 / P1=0 / P2=2
- Spec 偏离：minor=1 / major=0
- ⚠️ **强制提示（R-TEST-03）**：75k 集成未在提交态重跑
- ⚠️ unit 失败是 `http://ggml.ai/` 下载断言，不是 spec-chain 图/CLI 回归

## A. 总览【必填】

- Scope：`adf41b6b2` 相对其父 11 源文件 +934/−15（含事后 spec）；`7ca1534e9` 只回写 handoff SHA
- 覆盖维度：§D 约束逐条、plan §B vs diff、CLI 陷阱、logits 布局、双模型 unroll、verify 是否打中根因



## B. verify_informed 核心判定

- **compile 打中本需求源**：ninja 重配日志 `ggml commit: 7ca1534e9`；exe 能加载 CUDA + alderlake 并进入 `common_params_parse`
- **unit 未打中根因**：目标二进制的失败点是既有 HTTP 段（`test-arg-parser.cpp:187`），`--spec-chain` 三种解析形态**没有**单测。不能把「target 跑过」读成「chain 图正确」
- **集成未确认**：会话 75k 预研（chain=2 ~+2%）不是本轮 verify 证据



## D. P2 建议项

- `common/arg.cpp:3715` — `--spec-chain 1` 走 `is_truthy`，只开开关、不设 `n_max`（默认 3）。spec 已写明，建议帮助文本直接写「`1`/`on` 不等于深度 1」
- `tests/test-arg-parser.cpp` — 未覆盖 `--spec-chain` 的 truthy / falsey / 数字 N；verify 只能证明链接与旧 parse 用例跑到了 download 段



## E. 需求完整度【必填】

- `1-change-plan.md §B`：11 个业务改动点均在 `adf41b6b2` 中（CLI / speculative chain_graph / cparams API / graph 比较 / 两套 DECODER_MTP / server `n_outputs_max`）
- `0-clarification.md`：`perf` 仅 §A/§B；吞吐目标待提交态 75k 确认
- `0-context-snapshot.md §D`：
  - Diff 限于声明的 11 文件 + 本 spec — 遵守（`testing/CMakeLists.txt` 未进 commit）✓
  - 未提交 `.cursor/` / 旧 spec 快照 / `ai_docs/` — 遵守 ✓
  - `n_outputs_max >= n_parallel * n_max` — `server-context.cpp:1261-1263` 遵守 ✓
  - `mtp_chain` logits 按 `ggml_nbytes` — `llama-context.cpp:1467` 与 `:1906-1908` 遵守 ✓



## SC. Spec Check Report



## Spec Check Report

- spec: 0-clarification.md ; 1-change-plan.md
- spec_type: change | change-plan
- level_run: L1+L2+L3
- result: PASS_WITH_WARNINGS
- summary: 计划内文件均落地；单测未覆盖 chain；集成未跑；澄清文首仍写「未跑 /ai-verify」（minor drift）



### Errors

（无）

### Warnings

- [R-QUAL-07] `perf` 澄清仍为 §A/§B 占位
- [R-PLAN-03] 澄清无 `suggested_tests`；plan §E 自补了可判定测试点
- [R-TEST-01] §E 静态 CLI 三种形态 / graph 比较键无对应单测名称
- [R-TEST-03] `integration: not_run` 且 Env 预判需要集成



### Passed

- [R-STRUCT-01/02/03] perf 结构、frontmatter、`work_type` 与 meta 一致
- [R-PLAN-01/02/04/05] 必填节齐全；条目含路径/函数；Env 已答；§A 有顺序 MTP vs chain 取舍
- [R-DIFF-01/03] 提交源码路径均在 plan §B；无文档过宽漏改
- [R-TEST-02] manual 75k 已标明勿期望单测全覆盖



## SR. Spec Reconciliation



### SR-1 [minor] `0-clarification.md` 文首

- **偏离描述**：澄清仍写「未跑 `/ai-verify`」；现已有 `3-verify-report.md` revision 1
- **判定理由**：事后补齐时的时间戳句子，不是需求理解错误
- **建议**：`--reconcile` 改成「verify revision 1：compile pass；unit 因 ggml.ai HTTP 失败（infra）」



## 待修复清单

- [ ] FIX-1 [P2] `common/arg.cpp:3715` 帮助文本标明 `1`≠深度 1 — 状态: open
- [x] FIX-2 [P2] `test-arg-parser` 补 `--spec-chain` 三种解析 — 状态: done（`1` 只开开关、`2` 设深度、`off` 关闭；download 非 200 跳过）