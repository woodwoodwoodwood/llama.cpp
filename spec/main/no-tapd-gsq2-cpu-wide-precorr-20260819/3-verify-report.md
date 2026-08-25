## 0. Summary Card【必读】
- compile: result=pass
- unit: pass=6 / fail=0 / result=pass
- integration: not_run（manifest `integration_test_requested=true`，75k `-cmoe` 需人工触发 + 人工判定；基线 ~30 t/s、plan 验收线 ≥35 t/s、go 阶段预研参考 37.97 t/s）
- 归因（若 fail）：（无）
- 建议下一步：`/ai-review --mode=verify_informed`（消费本报告 revision 1）

## A. Compile 结果【必填，前置于 Unit】
- 是否通过：是
- 命令：`cmake --build D:\Codes\cakejiang\llama.cpp\build-x64-windows-llvm-release --target test-gsq2-vec-dot -j 20`（vcvars64 x64；Ninja）
- 退出码：0；38 个构建步骤，无错误。因定向 stash 回退了工作区的 q3_K 任务改动（与本需求文件不相交），CUDA 源（convert/fattn/ggml-cuda/set-rows/cpy）被重编回提交态——构建产物与提交态完全对齐
- **隔离说明**：交接注意事项按 manifest note 执行——实现已提交为 `2b5cc1b63`（author lijiajieli / co-author cakejiang），工作区另一任务（q3_K CUDA 移植，8 文件）以 `git stash push -- <files>` 定向隔离；本需求 meta.yaml 的 hash 刷新属预期保留
- **构建完整性探针（对象文件级，本轮修正）**：DLL 导出表不含内部函数符号（上轮已观察到），改查编译单元 `.obj`：
  - `ggml-cpu-alderlake.dir/ggml-cpu/quants.c.obj` 定义 `T ggml_quantize_row_gsq2_act` / `T ggml_gsq2_act_row_size`（wide 实现在场）
  - `ggml-cpu-alderlake.dir/ggml-cpu/ggml-cpu.c.obj` 未定义引用 `U ggml_quantize_row_gsq2_act` / `U ggml_gsq2_act_row_size`（调度接线生效，非仅库内有定义）
  - `ops.cpp.obj` 只读段含两条 fattn GSQ2 assert 消息字符串（guard 编入）
- CMake 配置：`GGML_BACKEND_DL=ON`、`GGML_CPU_ALL_VARIANTS=ON`、Release；工具链 VS2022 + Clang/LLVM 19.1.5 + CUDA 12.4 + Ninja
- 警告：仅既有 CUDA 告警，无本轮新增

## B. Unit 测试结果【必填】
- 前置条件：Compile=pass，已执行
- 覆盖目标：`unit_test_targets=["test-gsq2-vec-dot"]`
- 执行命令：`bin\test-gsq2-vec-dot.exe -v`；退出码 0
- **测试环境有效性**：loaded CPU backend = alderlake，`AVX2=1`（265KF）——打到 AVX2 wide 内核，非 generic 假绿
- 用例明细（6/6 pass）：
  - `pack 0xE4 to_float` — ok（bit 序契约锁定）
  - `pack 0xE4 mul_mat` — ok（got=-0.565094 ref=-0.565094）
  - `mul_mat GSQ2 m=1 n=1 k=128` — ok（**原 5 用例不修改全绿**——change-plan §E 回归锚点达成）
  - `mul_mat GSQ2 m=8 n=1 k=128` — ok
  - `mul_mat GSQ2 m=4 n=3 k=256` — ok
  - `mul_mat_id GSQ2 m=8 n_as=2 k=256` — ok（**新增 MoE 路径用例**：2 专家轮询路由 × 3 token，对照侧 Q8_0 走 CPU backend `ggml_cpy`，遵循 ai_docs/testing-quantized-vec-dot.md）
- 精度证据（change-plan §E"精度"测试点）：全部用例在 1e-4/1e-5 容差内通过——浮点求和重排（预研 mean 3.4e-07 / worst 3.4e-05）被容差覆盖
- 失败详情：无

## C. Integration 测试结果
- `integration_test_requested=true`，本轮未触发（需人工 `marvis-local-bench` 75k `-cmoe` + 人工判定）
- 结果：`not_run`
- 验收线（change-plan §E）：decode ≥ 35 t/s（基线 ~30；go 预研 37.97 为工作区参考值，正式验收须在提交态跑——本轮提交态与预研实现一致，预期可达，但仍需实测确认）
- 是否阻塞：不阻塞 unit 通过后的状态推进；留给 verify_informed review 强制提示

## D. 环境信息【必填】
- 环境标识：Windows 本机（CodeBuddy 会话代行容器 verify，沿先例；非独立容器镜像）
- 执行的 commit：`2b5cc1b639a44d978287ea8bcc315934ce1e32ab`（= manifest `produced_by_commit`，F5 校验通过）
- handoff：消费前 `pending=true`，本次写入 `consumed_at`/`consumed_by_commit`
- 定向 stash 往返：`stash@{0}` push（q3_K 8 文件）→ 验证 → pop 恢复，q3_K 改动完整回归
- 探针方法学修正记录：ai_docs/verify-clean-tree-isolation.md 的"符号探针"以 DLL 为对象在本仓 DL 构建下不适用（内部符号不导出），本轮改用对象文件级探针（定义 + 引用双侧验证）——该修正建议由 ai-reflect 回流进知识库
