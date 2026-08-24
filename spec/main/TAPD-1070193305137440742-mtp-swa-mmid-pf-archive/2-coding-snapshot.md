## Summary Card【必读】
- 改动范围（implementation_scope）：`tools/server/server-context.cpp`（1 处 inline 护栏，+6 行）、`tests/CMakeLists.txt`（1 条 CTest 注册追加，+4 行）；共 2 个文件，无新建文件。
- 调用链摘要（call_chain_summary）：`server.cpp:394` → `server_context::load_model` → 内部 `Impl::load_model`（MTP 草稿 context 组装处新增护栏，`return false` 走既有失败路径，调用方无需改动）。`test-gsq2-vec-dot` 可执行文件不变，仅新增一条以 `GGML_MMID_PF=0` 环境变量运行的 CTest 用例。
- 兼容风险（compat_risks）：无数据兼容/灰度问题。行为变化点仅一处——`LLAMA_MTP_SWA=W` 且 `n_ubatch>W` 的非法组合，改动前静默启动，改动后启动失败并打印 `SRV_ERR`；两个开关默认关闭时行为完全不变。
- 测试前置（test_prep）：`tests/CMakeLists.txt` 新增的 CTest 依赖已构建的 `test-gsq2-vec-dot` 目标（未新增可执行文件）；无需下载模型、无需 GPU。
- 需回源（needs_reread）：无——本文件已自包含所有改动细节，容器侧无需回读 `1-change-plan.md` 即可验证。

## Verification Handoff【必填，本项目新增】
- 本地自检能力：**无编译条件 / 未验证**（用户明确指示本阶段不做本地编译和运行，见 `1-change-plan.md §Env`）。本字段仅供容器侧归因参考，**不能替代**容器侧的权威编译；容器 verify 必须自己重新编译一次。
- 增量单测范围（unit_test_targets）：`["test-gsq2-vec-dot", "test-gsq2-vec-dot-mmid-pf0"]`（后者是同一可执行文件的第二条 CTest 注册，仅环境变量不同）。容器侧只需构建 `test-gsq2-vec-dot` 一个目标，跑这两条 CTest 用例即可。
- 覆盖缺口对照：
  - `1-change-plan.md §E` 第 1 条（`LLAMA_MTP_SWA` 护栏生效）：**未覆盖，人工验证项**——用户已明确认领人工测试，不要求自动化单测（见 `0-clarification.md §B` 第 3 条 + 用户确认）。容器侧只需确认**编译通过**（护栏代码本身无编译错误即可，不要求容器跑启动测试）。
  - `1-change-plan.md §E` 第 2 条（`GGML_MMID_PF` 双 env 值一致性）：已覆盖，对应 `unit_test_targets` 两条用例。
- 是否需要集成测试：是；`pending_integration_decision`——第 1 条测试点需要真实拉起 server + 传入 speculative/MTP 相关参数，本仓库无现成的 MTP server 集成测试夹具，且用户已明确表示人工验证，不要求自动化，决策为"人工触发 + 人工确认结果回填 `3-verify-report.md §integration`"（按 F7 默认降级处理）。
- 已知不确定项（供容器侧失败归因参考）：
  - 护栏代码里的 `SRV_ERR` 宏用法（格式串 4 个 `%u` 参数）与文件内其它 `SRV_ERR` 调用风格一致，但未实际编译验证参数类型完全匹配（`w`/`nub` 均已显式 `(uint32_t)` cast），**可能导致编译失败**（格式串警告，低概率）。
  - `tests/CMakeLists.txt` 新增的 `llama_test(test-gsq2-vec-dot NAME test-gsq2-vec-dot-mmid-pf0)` 依赖 `llama_test` 宏对已存在 target 的复用，未实际跑过 CMake configure 验证语法，**可能导致 CMake configure 失败**（宏参数用法本身在文件里有类似先例，如 `test-tokenizer-0` 系列复用同一 target 注册多条 test，风险低）。
  - **⚠️ 工作区当前为 dirty state**：`produced_by_commit` 记录的 `58ccc2609f87e418f724c2e08b870caeeca811eb` 是本次改动**之前**的 `HEAD`（本次两处改动未提交）。若容器侧是基于 clone/独立工作树验证（而非直接复用同一工作区），会**拿不到**这两处未提交改动，`git rev-parse HEAD` 校验会"一致但内容缺失"，不是协议 F5 描述的典型"不一致"，需人工确认容器侧验证方式：若容器与编码侧共享同一工作区（本地文件系统直接跑），无影响；若容器侧走独立 clone，需先提交这两处改动才能让 `produced_by_commit` 有意义。

## 修复历史
（首次交接，无修复历史）
