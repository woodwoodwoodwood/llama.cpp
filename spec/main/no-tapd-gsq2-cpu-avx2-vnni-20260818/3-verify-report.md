## 0. Summary Card【必读】
- compile: result=pass
- unit: pass=5 / fail=0 / result=pass
- integration: not_run（manifest `integration_test_requested=true`，75k `-cmoe` 需人工触发 + 人工判定；历史证据见 §C）
- 归因（若 fail）：（无）
- 建议下一步：`/ai-review --mode=verify_informed`（消费本报告 revision 2）

## A. Compile 结果【必填，前置于 Unit】
- 是否通过：是
- 命令：`cmake --build D:\Codes\cakejiang\llama.cpp\build-x64-windows-llvm-release --target test-gsq2-vec-dot -j 20`（vcvars64 x64；Ninja）
- 退出码：0；关键产物：`bin\ggml-cpu-alderlake.dll` 与 `bin\test-gsq2-vec-dot.exe` 均重新链接
- **隔离说明（本轮关键差异）**：编码侧工作区含两批未提交改动（wide+precorr 内核替换、q3_K CUDA 移植），与本提交同文件。为保证验证对象为提交态，执行前 `git stash push -m "ai-verify isolation: ..."`，在 `d6ab3cd2d` 干净树上增量编译（94 个构建步骤，0 错误）；验证后 `git stash pop` 恢复
- **构建完整性探针**：对重链的 `ggml-cpu-alderlake.dll` 做 `llvm-nm --defined-only` 符号检查，无 `ggml_quantize_row_gsq2_act` / `ggml_gsq2_act_row_size` 等 wide 内核新增符号 → 确认验的是提交态 narrow 内核，非工作区未提交版本
- CMake 配置：`GGML_BACKEND_DL=ON`、`GGML_CPU_ALL_VARIANTS=ON`、`GGML_NATIVE=OFF`、Release（-O3）；ggml commit 线：`9faf2bdd7 → 8e1f4ff30 → d6ab3cd2d`（fix 与 spec 文档经 amend 并入同一提交，代码内容同 9faf2bdd7 谱系）
- 工具链：VS 2022 BuildTools 17.14（MSVC 14.44）+ Clang/LLVM 19.1.5 + CUDA 12.4 + Ninja
- 警告：仅既有 CUDA 告警（`mmvf.cu` unused `rawg`、C4819 代码页等），无本轮新增

## B. Unit 测试结果【必填】
- 前置条件：Compile=pass，已执行
- 覆盖目标：`2-handoff-manifest.json.unit_test_targets=["test-gsq2-vec-dot"]`（revision 2 handoff，非上一轮的空列表）
- 执行命令：`bin\test-gsq2-vec-dot.exe -v`
- 退出码：0
- **测试环境有效性**（FIX-3 引入的检测）：loaded CPU backend = `ggml-cpu-alderlake.dll`，`AVX2=1`（Intel Core Ultra 7 265KF）→ 5 条用例全部打到 AVX2 优化核，非 generic-vs-generic 假绿
- 用例明细（5/5 pass）：
  - `pack 0xE4 to_float` — ok（LSB-first bit 序锁定）
  - `pack 0xE4 mul_mat` — ok（got=-0.565094 ref=-0.565094）
  - `mul_mat GSQ2 m=1 n=1 k=128` — ok
  - `mul_mat GSQ2 m=8 n=1 k=128` — ok
  - `mul_mat GSQ2 m=4 n=3 k=256` — ok
- 对照路径（FIX-2）：参考侧 Q8_0 量化经 CPU backend `ggml_cpy(F32→Q8_0)`，与 mul_mat 内部同一 `from_float`（x86 round-to-even），无 from_float_ref 分叉风险
- 失败详情：无

## C. Integration 测试结果
- `integration_test_requested=true`，本轮 **未触发**（需人工触发 `marvis-local-bench` 75k `-cmoe` + 人工判定）
- 结果：`not_run`
- 参考证据（非本轮验收）：历史会话 75k `-cmoe` decode ~30 t/s（narrow 内核，标量基线 ~16）；澄清目标"约 40 t/s"未在本轮复测。注：2026-08-19 实测 37.97 t/s 来自**工作区未提交的 wide 内核**，属下一需求范围，**不得**计入本提交验收
- 是否阻塞：不阻塞 unit 通过后的状态推进；留给 verify_informed review 强制提示

## D. 环境信息【必填】
- 环境标识：Windows 本机（CodeBuddy 会话代行容器 verify，沿上一轮 `cursor-local-verify` 代行先例；非独立容器镜像）
- 执行的 commit：`d6ab3cd2d762cfa886a9b775a04d665ddee756f3`（= handoff `produced_by_commit`，F5 校验通过）
- 对照 handoff：一致；`handoff.pending` 消费前为 true，本次消费写入 `consumed_at`/`consumed_by_commit`
- stash 隔离往返：`stash@{0}` push（19:45）→ 验证 → pop（见 meta 状态历史），工作区未提交改动完整恢复
