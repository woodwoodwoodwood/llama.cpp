## Summary Card【必读】
- 改动范围（implementation_scope）：`1-change-plan.md §B` 全部 6 个改动点——`quants.h`（`block_gsq2_act` + 2 声明）、`quants.c`（`ggml_quantize_row_gsq2_act` + row_size + generic 重写）、`arch/x86/quants.c`（wide 内核 `ggml_vec_dot_gsq2_q8_0` + `gsq2_dpbusd_epi32`，删 `bytes_from_2bit_32`）、`ggml-cpu.c`（五处接线：mul_mat/mul_mat_id 转换循环、两个 one_chunk、graph_plan 两处 wsize、llamafile 跳过）、`ops.cpp`（fattn GSQ2 K/V guard）、`tests/test-gsq2-vec-dot.cpp`（新增 `run_cpu_mul_mat_id_gsq2` + `test_mul_mat_id_matches_generic`，main 挂接）
- 调用链摘要（call_chain_summary）：F32 激活 → `ggml_quantize_row_gsq2_act`（内部调派发 `quantize_row_q8_0`，量化字节与标准 Q8_0 逐位一致）→ wdata `block_gsq2_act`（192B/128 元素）→ mul_mat/mul_mat_id one_chunk 经 traits 调 wide 内核（4 dpbusd + 1 fmadd/块，VNNI 三分支）→ `(sum·ds + corr)·d0`。fattn 路径显式 assert 拒绝 GSQ2 K/V；llamafile sgemm 跳过 GSQ2
- 兼容风险（compat_risks）：无数据格式变更（`block_gsq2` / GGUF / CUDA 不动；act 布局仅 wdata）。两处行为变化：① GSQ2 权重 + 预量化 Q8_0 src1 直通 → 显式 `GGML_ASSERT(F32)`（原为静默按错布局解释，实际推理激活恒为 F32，不受影响）；② CPU fattn GSQ2 K/V → 显式 assert（原 narrow 下数值碰巧正确，wide 下必错，拒绝优于静默错算）
- 测试前置（test_prep）：VS2022+Clang+Ninja 增量编译（本会话已跑通）；测试需 CPU 变体 DLL 与 exe 同目录（`build-x64-windows-llvm-release/bin`）
- 需回源（needs_reread）：`1-change-plan.md` §B（6 改动点逐条）与 §E（8 测试点）；`0-context-snapshot.md` §B（5 条编码约束，尤其 fattn guard 与单一激活布局）

## Verification Handoff【必填】
- 本地自检能力：有编译条件，**已过编译 + 已跑测试**（此字段不替代容器侧权威编译）：
  - `cmake --build ... --target test-gsq2-vec-dot`：14 个 CPU 变体重编 `ops.cpp`，0 错误
  - `test-gsq2-vec-dot -v`：**6/6 ok**（原 5 用例不修改全绿 + 新增 mul_mat_id 用例 ok），loaded backend = alderlake `AVX2=1`
- 增量单测范围（unit_test_targets）：`test-gsq2-vec-dot`
- 覆盖缺口对照（change-plan §E）：
  - [compile] target 编译 + 变体链接 → **本会话已覆盖**（14 变体 ops.cpp 重编；quants/h 改动的符号经 DLL 链接成功间接证实；verify 复核）
  - [数值-回归] 现有 5 用例不修改全绿 → **本会话已覆盖**
  - [数值-mmid] mul_mat_id 用例 → **本会话已覆盖**（新增用例 ok）
  - [静态] fattn guard 存在 → **本会话已落码**（ops.cpp assert + 注释；verify 检索复核）
  - [精度] 重排差异在容差内 → 现有用例容差即证据（1e-4/1e-5 全绿）
  - [缺口声明] `test-quantize-fns` DL 不编 → 沿用既有声明
  - [integration] 75k ≥ 35 t/s → 未跑，`pending_integration_decision`
  - [integration] 500 token 可读性 → 未跑
- 是否需要集成测试：是；标注 `pending_integration_decision`（需人工触发 + 人工判定；预研参考值 37.97 t/s，基线 ~30）
- 已知不确定项（供容器侧失败归因参考）：
  - **⚠️ 提交状态（本轮最重要的交接注意事项）**：实现当前为**工作区未提交改动**，HEAD `8532bda43` 不含本需求代码。**不能**照搬上一需求的"stash + 干净树"隔离法——会把实现本身 stash 掉。正确顺序：① 先把本需求 6 文件（+ 关键 spec 产物，沿用既有提交策略）提交为独立 commit；② 刷新本 manifest 的 `produced_by_commit` 为该 commit；③ 再按 `ai_docs/verify-clean-tree-isolation.md` 正常隔离验证（此时干净树=含实现的提交态）。注意工作区同时叠加 q3_K CUDA 移植（另一独立任务）的未提交改动——提交本需求时**必须排除**（`common/arg.cpp`、`ggml-cuda/*`、`testing/CMakeLists.txt`、`tools/llama-bench/*`）
  - 非 AVX2 变体（sse42/x64 等）走 generic 回退消费新布局：本会话仅 alderlake 实测；其余变体靠编译通过 + generic 逻辑与 AVX2 同布局（可能导致非 AVX2 平台测试失败，本机不可复现，属环境边界）
  - 精度重排为 ulp 级（预研 mean 3.4e-07 / worst 3.4e-05，在 1e-4 容差内）——若 verify 阈值收紧需回看 `0-clarification.md §B` 数值语义共识（已确认可接受）
