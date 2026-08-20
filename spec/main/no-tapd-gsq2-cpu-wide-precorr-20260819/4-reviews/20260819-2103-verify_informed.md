# no-tapd-gsq2-cpu-wide-precorr-20260819 — Review 报告

> Source: conversation  |  Subdomain: main  |  Complexity: M
> Date: 2026-08-19 21:03

## 0. Summary Card【必读】
- review_mode: verify_informed
- model: codebuddy-glm-5.3（⚠️ 协议偏离：`meta.yaml.models.review=opus`，本执行环境无法切换会话模型；与既往各轮同样以显著标注方式继续。如需严格合规，可在 Cursor 以 opus 复跑）
- 基于 3-verify-report.md revision 1（compile=pass, unit=pass 6/6, integration=not_run）
- Findings：P0=0 / P1=0 / P2=1
- Spec 偏离：minor=0 / major=0
- ⚠️ **强制提示（R-TEST-03）**：`integration: not_run` —— 75k `-cmoe` 验收线（≥35 t/s）与 500 token 文本回归**均未确认**，见 §C

## A. 总览【必填】
- Scope：`2b5cc1b63` vs `8532bda43`，9 文件 / +450 -79（wide 代码 6 + spec 关键产物 3；q3_K 任务 8 文件已按交接说明排除在提交外）
- 覆盖维度：verify_informed 三问（验证打中根因 / 验证方法自身有效性 / 集成确认状态）+ §D 约束清单逐条核对 + L1/L2/L3 spec 检查

## B. verify_informed 核心判定：验证是否打中根因
- **fattn guard（设计阶段的关键修正）— 证实编入**：`ops.cpp.obj` 只读段含两条 GSQ2 assert 消息字符串（k/v 各一）。这是把"第三条 vec_dot 消费路径静默错算"风险关闭的直接证据，非仅"代码里写了"
- **MoE 路径覆盖（上轮需求遗留缺口）— 补齐**：新增 `mul_mat_id GSQ2 m=8 n_as=2 k=256` 用例 ok；对照侧 Q8_0 走 CPU backend `ggml_cpy`（遵循 ai_docs/testing-quantized-vec-dot.md，无 from_float_ref 分叉假失败）
- **调度接线生效 — 双侧符号证据**：`ggml-cpu.c.obj` 含 `U ggml_quantize_row_gsq2_act` / `U ggml_gsq2_act_row_size` 未定义引用 + `quants.c.obj` 含 `T` 定义——证明 wide 路径真实接线，排除"实现存在但未启用"的假验证
- **验证方法自身缺陷的当轮识别（本轮亮点）**：交接说明指定的 DLL 级符号探针在本仓 DL 构建下**假阴性**（内部函数不进 DLL 导出表，上轮以"查无 wide 符号"通过是否定性检查属方法巧合），本轮当场升级为对象文件级探针（定义+引用双侧）。该缺陷属知识库文档错误，已列 §C 建议由 reflect 回流修正
- **增量编译一致性说明**：verify 构建 38 步中 `ops.cpp` 未重编（go 阶段已按相同内容编译、提交前后无编辑），ninja 增量语义有效；q3_K 相关 CUDA 源因定向 stash 回退而重编——构建产物与提交态一致

## C. P1 应改项
（无）

## D. P2 建议项
- [R-TEST-03 强制提示] **集成与性能验收未确认**：`integration: not_run`
  - 验收线：75k `-cmoe` decode ≥ 35 t/s（基线 ~30）。go 阶段预研 37.97 t/s 出自与本提交实现一致的工作区代码，预期可达但**不构成验收证据**
  - 建议：合 MR 前对提交态跑一次 75k bench（人工触发 `marvis-local-bench`）或明确接受预研值；500 token 可读性回归同理
- [info] `ai_docs/verify-clean-tree-isolation.md` 探针方法需修正：DLL 导出表不含内部符号（DL 构建特性），应改为对象文件级探针（`llvm-nm` 查编译单元 `.obj`，定义 T + 引用 U 双侧）——建议由本需求 `/ai-reflect` 提案回流，本轮已在 verify 报告 §D 记录方法学

## E. 需求完整度【必填】
- `1-change-plan.md §B` 改动点覆盖：6/6 已实现（quants.h / quants.c / arch-x86 / ggml-cpu.c 五处接线 / ops.cpp guard / 测试追加）
- `0-clarification.md §E/§F`：`work_type=perf` 占位（§A/§B）；§B 目标（~30 → ~38 t/s）达成情况待集成确认，数值语义共识（int 域位级一致 + ulp 级重排）已由单测容差覆盖
- `0-context-snapshot.md §D` 约束清单核对（提交态 2b5cc1b63）：
  - Diff 限于声明的六文件 + spec 关键产物（3 份，沿用既定提交策略）— 遵守；q3_K 的 `testing/CMakeLists.txt` 等未混入 ✓
  - 不改 `block_gsq2` / 量化公式 / GGUF / CUDA / traits 对外语义 — 遵守（`ggml-quants.c` 权重格式文件、`ggml-cuda/`、GGUF 均未触及；traits 表项未动）
  - 非 AVX2 generic 回退消费同一 `block_gsq2_act` 布局 — 遵守（generic 与 AVX2 同布局重写）
  - 两处行为变化（预量化 Q8_0 直通 assert / fattn GSQ2 拒绝）均已落地并在 coding-snapshot 记录理由 — 遵守
  - 现有 5 用例不修改全绿 — 遵守（6/6，原 5 用例零改动）
  - mul_mat_id 对照测试 — 遵守（新增用例）

## SC. Spec Check Report

## Spec Check Report
- spec: 0-clarification.md ; 1-change-plan.md
- spec_type: change | change-plan
- level_run: L1+L2+L3
- result: PASS_WITH_WARNINGS
- summary: 六改动点全落地且验证证据链闭合（实现存在/接线生效/guard 编入三重探针）；集成未跑（强制提示）；知识库探针方法缺陷待 reflect 回流

### Errors
（无）

### Warnings
- [R-QUAL-07] `perf` 澄清为 §A/§B 占位（沿用）
- [R-PLAN-03] 澄清无 `suggested_tests`；plan §E 自补 8 条可二元判定测试点（沿用）
- [R-TEST-03] `integration: not_run` 且 Env 预判需集成 → §0/§C 强制提示

### Passed
- [R-STRUCT-01/02/03] perf 类型结构合规；frontmatter 齐全；work_type 一致
- [R-PLAN-01/02/04/05] plan 必填节齐全、条目含路径/函数名/条件；Env 已答；§A 含方案取舍（恒等式 + 2-row 比选）
- [R-DIFF-01/03] 提交 diff 全部路径在 plan §B 声明范围内；无文档过宽/漏改
- [R-TEST-01] §E 测试点证据映射：compile 点 ✓（38 步 0 错）；静态点 ✓（guard 字符串探针）；数值-回归点 ✓（原 5 用例零改动全绿）；数值-mmid 点 ✓（新用例 ok）；精度点 ✓（容差覆盖）；缺口点 ✓（quantize-fns 不编如实声明）；manual 点未跑（见 R-TEST-03）
- [R-TEST-02] manual-only 点已提示勿期望单测全覆盖

## SR. Spec Reconciliation
本次无 spec 偏离。

## 待修复清单
（无 open 项；本需求尚无前序 review 轮次）
