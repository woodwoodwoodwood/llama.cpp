## Summary Card【必读】
- 改动范围（implementation_scope）：核仍是 `9faf2bdd7`（`arch/x86/quants.c` + `arch-fallback.h` x86 去 alias）。本轮 `/ai-go --redo` 只补单测：新增 `tests/test-gsq2-vec-dot.cpp`，`tests/CMakeLists.txt` 在 `GGML_BACKEND_DL` 门外注册该 target，`tests/test-quantize-fns.cpp` 给 GSQ2 加上 2-bit 误差档。
- 调用链摘要（call_chain_summary）：单测经 `ggml_backend_load_all` → CPU `ggml_mul_mat(GSQ2, F32)` → `type_traits_cpu[GSQ2].vec_dot`（本机 alderlake AVX2/VNNI）。对照是测试内 `vec_dot_gsq2_q8_0_ref`（复刻 generic LSB-first / `code-2`）。`0xE4` 用例另走 `ggml_get_type_traits(GSQ2)->to_float`。
- 兼容风险（compat_risks）：不改 `block_gsq2` / GGUF / CUDA。风险仍是 unpack 与 generic 不一致；现由单测锁 bit 序与 mul_mat 数值。
- 测试前置（test_prep）：编并跑 `test-gsq2-vec-dot`（需 CPU backend DLL 与 exe 同目录）。`test-quantize-fns` 在本机 DL 构建下仍不编。75k bench 仍需 GSQ2 GGUF。
- 需回源（needs_reread）：`1-change-plan.md` revision 2 的 §B/§E/§Env。

## Verification Handoff【必填】
- 本地自检能力：有编译条件。本会话已 `ninja -C build-x64-windows-llvm-release test-gsq2-vec-dot`（CMake 因 `tests/CMakeLists.txt` 变动 re-run）并运行 `bin/test-gsq2-vec-dot.exe`，5 条用例均 ok。此字段不能替代容器侧权威编译。
- 增量单测范围（unit_test_targets）：`test-gsq2-vec-dot`。
- 覆盖缺口对照（change-plan §E）：
  - [unit / compile] 编出 `test-gsq2-vec-dot` → **本会话已覆盖**（exit 0）。
  - [unit / 静态] x86 `#elif` 无 GSQ2 generic alias → **本会话未重核**，源码相对 `9faf2bdd7` 未改 `arch-fallback.h`。
  - [unit / 数值] 跑 `test-gsq2-vec-dot` → **本会话已跑过且通过**；verify 须自己再跑一遍。
  - [unit / 缺口] `test-quantize-fns` 本机不编 → **确认仍不在 unit_test_targets**。
  - [integration / 人工] 75k `-cmoe` → 未在本轮重跑，`pending_integration_decision`。
  - [integration / 回归] 500 token 抽检 → 未重跑。
- 是否需要集成测试：是；标注 `pending_integration_decision`。
- 已知不确定项：
  - 对照用 `from_float_ref` 打 Q8；若 CPU mul_mat 内部 Q8 与 ref 分叉，单测会失败（更可能是测试假设，不是核错）。
  - 工作区另有未提交的 `testing/CMakeLists.txt`，**不在** §B；verify 勿当成本需求 diff。
  - 本交接 `produced_by_commit=9faf2bdd7`（HEAD 未变；单测仍是未提交工作区改动）。若之后只提交 `spec/` 导致 HEAD 变化，verify 按 F5 拒绝或先更新 handoff。

## 修复历史
- revision 2：针对 `3-verify-report.md` revision 1 的 **spec_drift / unit 缺口**（vacuous pass：`unit_test_targets=[]`），按 change-plan revision 2 补上 `test-gsq2-vec-dot`，不再空列表。
