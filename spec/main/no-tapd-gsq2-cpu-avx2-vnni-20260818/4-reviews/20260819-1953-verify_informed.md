# no-tapd-gsq2-cpu-avx2-vnni-20260818 — Review 报告

> Source: conversation  |  Subdomain: main  |  Complexity: M
> Date: 2026-08-19 19:53

## 0. Summary Card【必读】
- review_mode: verify_informed
- model: codebuddy-glm-5.3（⚠️ 协议偏离：`meta.yaml.models.review=opus`，本执行环境无法切换会话模型；与上轮 go_only 同样以显著标注方式继续。如需严格合规，可在 Cursor 以 opus 复跑）
- 基于 3-verify-report.md revision 2（compile=pass, unit=pass, integration=not_run）
- Findings：P0=0 / P1=0 / P2=1
- Spec 偏离：minor=0 / major=0
- ⚠️ **强制提示（R-TEST-03）**：`integration: not_run` —— 75k `-cmoe` 端到端性能与 500 token 文本可读性回归**均未确认**，见 §C

## A. 总览【必填】
- Scope：`d6ab3cd2d` vs `325fa87b9`，10 文件 / +797 -2（代码 4 + spec 关键产物 4 + 测试 2）
- 覆盖维度：verify_informed 专属三问（验证是否打中根因而非掩盖症状 / 失败归因有无隐患 / 验证环境有效性）+ 上轮 FIX-1~4 的证据复核 + §D 约束清单 + L1/L2/L3 spec 检查

## B. verify_informed 核心判定：验证是否打中根因
- **FIX-1（P1，阈值）— 正确分类，非掩盖**：根因是 GSQ2 量化格式的固有属性（`d=amax/2` + 非对称 codebook `{-2,-1,0,1}` 正侧只覆盖到 `+amax/2`），不是内核错误。改 `quantize_row_gsq2_ref` 会破坏 GGUF 二进制兼容（§B 明确禁止），独立阈值带 + 成因注释是唯一正确路径
  - 证据边界（如实记录）：本轮 verify 为 DL 构建，`test-quantize-fns` 未编（plan §E 已声明该缺口）；阈值有效性证据来自评审轮独立非 DL 构建：abs RMSE 0.00796 < 0.010、dot 0.213 < 0.25
- **FIX-2（对照同源）— 验证证实有效**：verify 报告的参考侧走 CPU backend `ggml_cpy(F32→Q8_0)`，与 mul_mat 内部同一 `from_float`；5 用例数值全绿且无假失败来源
- **FIX-3（backend 检测）— 验证证实有效**：verify 输出实际打印 `CPU backend: ... 265KF AVX2=1`，证明检测逻辑真实生效而非摆设；加载的是 alderlake 变体（AVX2+VNNI 优化核）
- **FIX-4（unaligned load）— 编译证实**：干净树重编 0 错误；`memcpy` 模式与同文件 helper 一致
- **验证环境有效性**（防"假绿"三要素）：干净树（stash 隔离未提交改动，F5 commit 校验过）✓；构建完整性（符号探针确认 DLL 无 wide 内核符号）✓；AVX2 backend 真实加载 ✓

## C. P1 应改项
（无）

## D. P2 建议项
- [R-TEST-03 强制提示] **集成与性能目标未确认**：`integration: not_run`
  - 澄清 §B 目标"约 40 t/s（对齐同机 IQ2_M）"未被提交态代码证实：历史 narrow 内核 75k decode ~30 t/s（vs 标量基线 ~16，**已获 +87%**，但未达 40 t/s 目标值）
  - 2026-08-19 实测 37.97 t/s 出自**工作区未提交的 wide 内核**，属下一需求范围，不得计入本提交验收
  - 建议：人工决策——(a) 接受 ~30 t/s 作为本提交验收值（目标值留给 wide 内核需求闭环）；或 (b) 补跑提交态 75k bench 后再合 MR。500 token 文本可读性回归同理未跑
- [R-TEST-01 边界记录] `test-quantize-fns`（FIX-1 阈值的执行载体）在 DL 构建下不可编：plan §E 已声明、不升 finding；若后续有非 DL CI，首次跑即为阈值的首个自动化证据

## E. 需求完整度【必填】
- `1-change-plan.md §B` 改动点覆盖：5/5 已实现
- `0-clarification.md §E/§F`：`work_type=perf` 占位类型（§A/§B），目标值达成情况见 §D 第一条
- `0-context-snapshot.md §D` 约束清单核对（提交态 d6ab3cd2d，与 go_only 轮相同结论）：
  - 核 diff 限于 `arch/x86/quants.c` + `arch-fallback.h` x86 分支 — 遵守
  - 未改 `block_gsq2` / quantize / dequant / CUDA / traits — 遵守
  - 非 AVX2 回退 generic — 遵守
  - 非 x86 fallback alias 未删 — 遵守
  - `quants.h` 签名未改 — 遵守
  - 测试三文件范围 — 遵守

## SC. Spec Check Report

## Spec Check Report
- spec: 0-clarification.md ; 1-change-plan.md
- spec_type: change | change-plan
- level_run: L1+L2+L3
- result: PASS_WITH_WARNINGS
- summary: 单测证据链完整且验证环境有效性三要素齐备；集成未跑（强制提示），性能目标值未被提交态证实

### Errors
（无）

### Warnings
- [R-QUAL-07] `perf` 澄清为 §A/§B 占位（沿用）
- [R-PLAN-03] 澄清无 `suggested_tests`；plan §E 自补 6 条测试点（沿用）
- [R-TEST-03] `integration: not_run` 且 Env 预判需集成测试 → 本报告 §0/§D 已强制提示

### Passed
- [R-STRUCT-01/02/03] perf 类型结构合规；frontmatter 齐全；work_type 一致
- [R-PLAN-01/02/04/05] plan 必填节齐全可执行；Env 已答自检/容器问题；§A 有选型理由
- [R-DIFF-01/03] diff 路径全部在 plan §B 声明范围内；无文档过宽/漏改
- [R-TEST-01] §E unit 测试点证据映射：compile 点（llama-server 于 verify r1、test target 于 r2）✓；静态点（x86 alias 检索，go_only 轮）✓；数值点（5/5，本轮）✓；缺口点（quantize-fns 不编，如实声明）✓
- [R-TEST-02] manual-only 点（75k、500 token）已提示勿期望单测全覆盖

## SR. Spec Reconciliation
本次无 spec 偏离。

## 待修复清单
（本模式无 open 项；上轮 go_only 的 FIX-1~4 已全部 resolved 并经本轮 verify 证据复核）
