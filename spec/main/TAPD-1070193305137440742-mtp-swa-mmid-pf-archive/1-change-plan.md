> 复杂度声明：`meta.yaml.complexity=M` 维持不变。文件数维度触及 3 个文件，改动性质（局部行为微调 + 补测试）/协作范围（单仓单人）/兼容性（无负担，两开关默认关闭不变）均落在 M。

### A. 问题分析与方案总述【必填】
- **问题分析**：两个已提交特性（`1b32c65da` MTP_SWA、`00f897882` MMID_PF）代码本身完整且有实测数据，唯一的真实缺口是 `0-clarification.md §B` 第 3 条确认的运行时护栏：`server-context.cpp` 组装 MTP 草稿 `cparams_mtp` 时读取 `LLAMA_MTP_SWA`，注释写 `Requires n_ubatch <= W` 但没有代码强制这条约束——`n_ubatch>W` 时窗口容量 `2*W*n_parallel` 无法覆盖一个在途 ubatch，存在静默越界风险。`GGML_MMID_PF` 侧不存在护栏缺口，但缺自动化测试覆盖。
- **代码定位复核（用户要求核实无遗漏）**：`load_model` 的唯一调用方是 `server.cpp:394`（`if (!ctx_server.load_model(params)) { ...; return 1; }`），新增护栏走同一 `return false` 失败路径，与既有的 `ctx_dft == nullptr` 等失败分支语义完全一致，调用方无需改动、无特殊处理需要担心。`params_base.n_ubatch`（`int32_t`）、`params_base.n_parallel`（`int32_t`，默认 1）均已在同一函数作用域内可直接访问，无需额外传参。未发现其它遗漏改动点。
- **方案总体思路**：
  1. 护栏检查直接 inline 写在 `server-context.cpp` 现有 `if (const char * e = std::getenv("LLAMA_MTP_SWA"))` 块内，不额外抽取纯函数（用户明确要求："不要单独为测试而抽取"）——护栏本身逻辑足够简单（一次比较 + 一条错误日志），跟随现有代码风格原地实现即可。
  2. `src/llama-model.cpp` 中另一处独立的 `std::getenv("LLAMA_MTP_SWA")` 读取**不改动**（不属于本次确认范围，两处解析同一环境变量目前产出一致结果，非本轮回归风险，记入 §G）。
  3. `LLAMA_MTP_SWA` 护栏的验证方式为**人工测试**（用户明确认领，不需要自动化单测，也不需要在文档中特别加注说明）。
  4. `GGML_MMID_PF` 侧不新增代码，只补测试：现有 `tests/test-gsq2-vec-dot.cpp::test_mul_mat_id_matches_generic` 已经用真实 `GGML_TYPE_GSQ2` 走 `ggml_mul_mat_id`（触发 `ggml-cpu.c` 里 `GGML_MMID_PF` 生效的同一代码路径）并对照 `vec_dot_gsq2_q8_0_ref` 做数值校验；缺的只是**在不同 `GGML_MMID_PF` 取值下都跑一次**。由于 `ggml_mmid_pf_distance()` 用 `static int d = -1` 做进程内一次性缓存，同一进程无法先测默认值再测 `0`，因此用 CTest 把同一个 `test-gsq2-vec-dot` 二进制注册成两条独立测试用例（默认 env / `GGML_MMID_PF=0`），各自独立进程运行，两者都必须 PASS。不新建测试文件，不改 `test-gsq2-vec-dot.cpp` 本身。

### B. 代码定位【必填】

**改动点 1 — MTP_SWA 护栏（inline）**
- `tools/server/server-context.cpp:1269-1280`（现有 `if (const char * e = std::getenv("LLAMA_MTP_SWA")) { uint32_t w = ...; if (w > 0) { ... } }` 块）：在 `if (w > 0) {` 内、计算 `cap` 之前插入：
  ```cpp
  uint32_t nub = (uint32_t) params_base.n_ubatch;
  if (nub > w) {
      SRV_ERR("LLAMA_MTP_SWA=%u requires n_ubatch <= %u, got n_ubatch=%u\n", w, w, nub);
      return false;
  }
  ```
  其余既有逻辑（`np`/`cap` 计算、`cap < cparams_mtp.n_ctx` 判断、`SRV_INF` 日志、赋值）不变。

**改动点 2 — MMID_PF 双 env 值 CTest 注册**
- `tests/CMakeLists.txt:244`（`llama_build_and_test(test-gsq2-vec-dot.cpp)` 所在行）之后追加：
  ```cmake
  llama_test(test-gsq2-vec-dot NAME test-gsq2-vec-dot-mmid-pf0)
  set_tests_properties(test-gsq2-vec-dot-mmid-pf0 PROPERTIES ENVIRONMENT "GGML_MMID_PF=0")
  ```
  复用同一 `TEST_TARGET`（`test-gsq2-vec-dot`），原有默认注册不变（隐式验证默认 `GGML_MMID_PF`，未设置回落 8，同样 PASS）。

### C. 调用链影响【必填】
- 改动点 1：唯一调用方 `server.cpp:394`，已复核（见 §A），无需扩大排查范围。
- 改动点 2：纯 CMake 测试注册，不涉及生产代码调用链。

### D. 数据兼容
（无变更：不涉及数据格式/GGUF/存储结构，整节删除）

### E. 测试要点【必填】
1. **`LLAMA_MTP_SWA` 护栏生效**（人工验证）：用 `-np 1 --spec-draft-n-max 2 --ubatch-size <大于W>` 加 `LLAMA_MTP_SWA=<小W>` 启动 server，预期启动失败并打印 `SRV_ERR` 信息；正常组合（`n_ubatch<=W`）预期行为与改动前一致（沿用 commit 里的实测数据为证据，不重新采集）。
2. **`GGML_MMID_PF` 不同取值下 `mul_mat_id` 数值一致**（unit，`test-gsq2-vec-dot` 的两条 CTest 注册均需 PASS，对照同一份 `vec_dot_gsq2_q8_0_ref` 参考实现）——覆盖默认（回落 8）与显式 `0`（关闭）两态。

### F. 灰度回滚
- 两个开关本身默认关闭（不设置对应环境变量），行为与改动前完全一致，无需灰度。
- 新增护栏是**行为变化点**：之前 `LLAMA_MTP_SWA` + `n_ubatch>W` 的非法组合会静默启动（潜在越界风险）；改动后会显式拒绝启动。回滚方式：`git revert` 改动点 1 即可，不影响改动点 2。

### G. 风险待办【必填，可为空】
- `src/llama-model.cpp` 中存在第二处独立的 `std::getenv("LLAMA_MTP_SWA")` + `std::atoi` 读取（用于设置 KV cache 的真实 `n_swa`），与 `server-context.cpp` 的解析逻辑重复但目前结果一致；本轮不合并，仅记录不处理。
- 改动点 1 新增了一种 `return false` 原因，调用方 `server.cpp:394` 统一按"模型加载失败"处理，语义一致，风险低。

### Env 执行环境提示【必填，本项目新增】
- 本次改动能否在编码环境内做语法/类型级自检？**本轮不做**（用户明确指示："不用，不用编译和运行"）；`ai-go` 阶段按 F4 诚实标注"未尝试本地验证"，不编造已编译/已验证通过，真实验证交给容器侧 `ai-verify`。
- 是否涉及需要容器内才能验证的行为？是——改动点 1/2 均需要实际编译链接 `server-context`/`llama`/`llama-common` 相关目标。
- 建议的验证范围（增量单测目标）：`test-gsq2-vec-dot`（含新增的 `test-gsq2-vec-dot-mmid-pf0` CTest 变体）；`server-context.cpp` 的护栏改动无独立单测目标，靠 compile 通过 + 人工验证。
- 是否预判需要集成测试：**是**，§E 第 1 条（server 真实启动路径的护栏生效）需要人工触发，用户已明确认领人工测试，`3-verify-report.md §integration` 可标 `not_run` 并说明"人工验证，非自动化"。
