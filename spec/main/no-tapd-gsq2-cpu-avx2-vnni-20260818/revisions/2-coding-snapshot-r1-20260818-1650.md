## Summary Card【必读】
- 改动范围（implementation_scope）：change-plan §B 两点均已在 `9faf2bdd7` 落地，本轮 `/ai-go` **未再改业务代码**。文件：`ggml/src/ggml-cpu/arch/x86/quants.c`（`bytes_from_2bit_32`、`ggml_vec_dot_gsq2_q8_0`）、`ggml/src/ggml-cpu/arch-fallback.h`（仅去掉 x86 的 GSQ2 generic alias）。
- 调用链摘要（call_chain_summary）：`type_traits_cpu[GGML_TYPE_GSQ2].vec_dot` → `ggml_vec_dot_gsq2_q8_0`（x86+AVX2 走向量核，否则 generic）。上层 mul_mat 经 type traits 间接调用。
- 兼容风险（compat_risks）：不改 `block_gsq2` / GGUF / CUDA。风险是 unpack 与 generic 不一致导致静默错算；无单测锁数值。
- 测试前置（test_prep）：compile `llama-server`（`GGML_CUDA=ON` + `GGML_CPU_ALL_VARIANTS=ON`）；无 GSQ2 CPU vec_dot unit。性能需本机 GSQ2 GGUF + `marvis-local-bench` 75k `-cmoe`。
- 需回源（needs_reread）：`1-change-plan.md` §B/§E/§Env；`0-context-snapshot.md` §B/§D。

## Verification Handoff【必填】
- 本地自检能力：有编译条件，**本会话未重新编译、未跑单测**。工作区当前未见 `build-x64-windows-llvm-release/bin/llama-server.exe`，不能把历史会话的链接成功写成「已编译通过」。容器 / verify 必须自己重新 compile（F4：不得用本字段替代权威编译）。
- 增量单测范围（unit_test_targets）：**无**。仓库没有 GSQ2×Q8 CPU `vec_dot` 对照用例；`unit_test_targets` 为空的理由：无需编造不存在的 target，verify 在 compile 通过后将 unit 记为 `not_run`。
- 覆盖缺口对照（change-plan §E）：
  - [unit / compile] 编 `llama-server` 且 x86 变体能链接 `ggml_vec_dot_gsq2_q8_0` → **未在本会话覆盖**，交 verify。
  - [unit / 静态] x86 `#elif` 无 GSQ2 generic alias、其它 ISA 保留 → **本会话已核对源码**：x86 块（`arch-fallback.h` L85–111）无该 `#define`；L20/L84/L118 等非 x86 仍有。
  - [unit / 缺口] GSQ2 vec_dot 对照单测 → **确认不存在**，不伪称覆盖。
  - [integration / 人工] 75k `-cmoe` decode ~28–31 t/s → **未在本会话重跑**，`pending_integration_decision`。
  - [integration / 回归] 500 token 抽检可读 → 未在本会话重跑。
- 是否需要集成测试：是；标注 `pending_integration_decision`（人工触发 `marvis-local-bench`，F7）。
- 已知不确定项：
  - unpack bit 序若与 `quantize_row_gsq2_ref` 不一致 → 可能导致测试/推理失败（静默错算），而非编译失败。
  - 工作区另有未提交的 `testing/CMakeLists.txt`，**不在** §B / 本 commit；verify 勿把它当本需求 diff。
  - 本交接 `produced_by_commit=9faf2bdd7`。若之后只提交 `spec/` 导致 HEAD 变化，verify 须按 F5 拒绝或先更新 handoff。

## 修复历史
（首次交接，无 `--redo`）
