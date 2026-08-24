## 0. Summary Card【必读】
- compile: result=pass
- unit: pass=0 / fail=1 / result=fail
- integration: not_run（manifest `integration_test_requested=true`；75k 人工 bench 本轮未重跑）
- 归因：`test_infra_issue`（`test-arg-parser` 在 parse 段之后访问 `http://ggml.ai/`，`res.first != 200`，与 spec-chain 无关）
- 建议下一步：`/ai-review --mode=verify_informed`（消费本报告 revision 1）。HTTP 失败不回 `ai-go`。parse 段 stdout 已出现预期的非法参数用法，未见 `--spec-chain` 引入的解析崩溃。

## A. Compile 结果【必填，前置于 Unit】
- 是否通过：是
- 命令：`ninja -C build-x64-windows-llvm-release test-arg-parser`（vcvars64 x64）
- 第一次：ninja 因 build log 版本过旧全量重配；CMake 日志 `ggml commit: 7ca1534e9`（= 当时 HEAD / 刷新后的 `produced_by_commit`）
- 第二次：`ninja: no work to do`（target 已在图中，可执行文件能启动）
- 隔离：`git stash push -- testing/CMakeLists.txt`（无关 CUDA bench-kernels），验证期间未混入该 diff
- 工具链：VS2022 BuildTools + Clang/LLVM + CUDA 12.4 + Ninja；本机编码环境代行容器 compile（无独立容器镜像，沿本仓库先例）
- 警告：既有 CUDA `mmvf.cu` unused `rawg`，非本需求

## B. Unit 测试结果【必填】
- 前置条件：Compile=pass，已执行
- 覆盖目标：`unit_test_targets=["test-arg-parser"]`
- 命令：`bin\test-arg-parser.exe`；退出码 `-1073740791`（abort）
- 加载：`ggml-cuda.dll` + `ggml-cpu-alderlake.dll`（RTX 5060 / 265KF）
- 已执行到 download 段之前的 parse 用例（stdout 含 `-m`/`-ngl`/`-sm`/`--draft`/`--model is required` 等**预期**失败用法）
- 失败详情：`tests/test-arg-parser.cpp:187` `assert(res.first == 200)` — `common_remote_get_content("http://ggml.ai/")`。该段不测 `--spec-chain`，Windows 上也不跳过
- 缺口（当时）：无 `--spec-chain` 三种形态断言（change-plan §E 已标）

## C. Integration 测试结果
- revision 1 未触发。结果：`not_run`
- 验收仍依赖人工 75k + `LLAMA_SPEC_CHAIN_SUB=0`；会话预研不构成本轮提交态验收
- 不把 integration 未跑写成 pass

## E. 后续（同一需求、verify r1 之后）
- `test-arg-parser` 已补 `--spec-chain` 三种解析，且 `ggml.ai` 非 200 时跳过 download，不再因 infra abort
- 75k `--spec-chain 2` + `SUB=0`：decode 58.48 t/s，accept 0.733，mean len 2.47
- 8k greedy 对齐：chain 与顺序 MTP 499 token 全等、accept 数字咬死；与关 MTP 在 token 13 分叉（既有 MTP 差异）

## D. 环境信息【必填】
- 环境标识：Windows 本机代行容器 verify
- HEAD / consumed（当时）：`7ca1534e9bd65377d0ae5907b49c846a269f4c23`；事后与实现 commit 折入 `d554ecb12c9503b26121c42a465c9f4ec944ef5e`
- F5：当时把 `produced_by_commit` 从 `adf41b6b2` 刷新为 `7ca1534e9`（仅 handoff SHA）后再编
- handoff：消费前 `pending=true`；本报告写入 `consumed_at` / `consumed_by_commit`
