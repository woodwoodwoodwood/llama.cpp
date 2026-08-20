### A. 问题分析与方案总述
- 问题分析：
  1. 现行 narrow 内核（`arch/x86/quants.c` `ggml_vec_dot_gsq2_q8_0`）每 128 元素块调 `bytes_from_2bit_32` ×4（~60 条指令：load 宽化 + 4 组 shift/and + 3 次拼回 + abs/sign 预处理），而实际乘加只需 4 条 dpbusd。预研实测：内核算力 ~40 GB/s，仅为 L3 常驻上限（65.6）的 61%，指令预算被解包吃掉。
  2. 消费方反搜（闭合 clarification §C 缺口 1，**假设修正**）：`type_traits_cpu[GSQ2].vec_dot` 的调用方共三处——`ggml_compute_forward_mul_mat_one_chunk` 与 `ggml_compute_forward_mul_mat_id_one_chunk`（`ggml-cpu.c`），以及 **CPU flash attention（`ops.cpp:8440-8442`，经 `kq_vec_dot = type_traits_cpu(k->type)->vec_dot`）**。fattn 用 `vec_dot_type`(Q8_0) 的 `from_float` 以**标准 q8_0 布局**量化 Q，直接喂新内核会静默错算 → 必须显式 guard（GSQ2 为权重专用格式，llama.cpp 侧 K/V 无 GSQ2 入口，但 ggml 图层面可达）。
  3. llamafile sgemm 路径（`ggml-cpu.c` 两处）也不理解自定义激活布局，须跳过 GSQ2。
- 方案总体思路：**wide 解包 + 激活置换 + int 域预计算修正**。数学依据 `sum((code-2)·y) = sum(code·y) − 2·sum(y)`：code 无符号 0..3 直接喂 dpbusd（免 abs/sign），`-2·sum(y)` 只依赖激活、在 prep 阶段算成 per-lane 向量。激活在 mul_mat/mul_mat_id 准备阶段由 `ggml_quantize_row_gsq2_act` 一次置换成 `block_gsq2_act` 布局（每 token 一次，被所有 expert 行摊销）；内循环降为 4 dpbusd + 1 fmadd/块。比选：曾评估 2-row 配对（`dot_wide_2row`，共享激活 load），本机（Arrow Lake 混合核）实测无增益，留作后续扩展，本轮不做。数值语义：整数域位级一致（int 域修正），仅浮点求和顺序重排（预研 200 组随机：mean rel err 2.3e-07→3.4e-07，均为 fp32 舍入级）。

### B. 代码定位
- `ggml/src/ggml-cpu/quants.h` — 新增 `block_gsq2_act`（int8_t qs[128] 置换值 + float d[8] lane scale + float corr[8] 预计算修正，192B/128 元素；仅 wdata 布局，不落盘）与 `ggml_gsq2_act_row_size(int64_t n)`、`ggml_quantize_row_gsq2_act(const float *, void *, int64_t)` 声明
- `ggml/src/ggml-cpu/quants.c` — ① 实现 `ggml_quantize_row_gsq2_act`：每 128 元素调**派发的** `quantize_row_q8_0`（保证量化字节与标准路径逐位一致）→ 按 `w=4i+k` 置写入 `qs[k*32+i]` → 组 `d[8]=[d0,d0,d1,d1,d2,d2,d3,d3]` → 累加 per-lane `sumy`、写 `corr[j]=-2·sumy[j]·d[j]`；② `ggml_vec_dot_gsq2_q8_0_generic` 重写为消费新布局的标量参考（按自然权重序 `w` 迭代，分组/缩放顺序与旧版一致）
- `ggml/src/ggml-cpu/arch/x86/quants.c` — `ggml_vec_dot_gsq2_q8_0` 替换为 wide 内核：32B `qs` 一次 load，`and(pk,3)` / `and(srli_epi16(pk,2|4|6),3)` 四向量对 `y0..y3` dpbusd；收尾 `(sum·ds + corr)·d0` fmadd。VNNI 三分支（`__AVX512VNNI__+VL` / `__AVXVNNI__` / maddubs 回退，新增 static helper `gsq2_dpbusd_epi32`）。删除已无使用者的 `bytes_from_2bit_32`
- `ggml/src/ggml-cpu/ggml-cpu.c` — 五处接线：① `mul_mat` 转换循环：`src0_is_gsq2` 时分流到 `ggml_quantize_row_gsq2_act`（块切分粒度改 128）；② `mul_mat_id` 转换循环：同①；③ 两个 `one_chunk`：`wdata` 来源与 `row_size` 对 GSQ2 取 `ggml_gsq2_act_row_size`（预量化 src1 直通路径判 `!is_gsq2`）；④ `graph_plan`：MUL_MAT/MUL_MAT_ID 两处 wsize 预留按 act 布局计；⑤ 第二处 llamafile sgemm 尝试跳过 GSQ2
- `ggml/src/ggml-cpu/ops.cpp` — CPU fattn 入口（`ggml_compute_forward_flash_attn_ext*` 取 traits 处，~L8440）加 `GGML_ASSERT(k->type != GGML_TYPE_GSQ2 && v->type != GGML_TYPE_GSQ2)`：把"理论上可达的静默错算"变为显式报错（行为变化：旧 narrow 内核下 GSQ2-K fattn 数值碰巧正确，新布局下必须拒绝）
- `tests/test-gsq2-vec-dot.cpp` — 追加 `test_mul_mat_id_matches_generic`：多专家（≥2）× 多 token × k≥256，权重经 `from_float_ref`、激活对照经 CPU backend `ggml_cpy(F32→Q8_0)`（遵循 ai_docs/testing-quantized-vec-dot.md），标量参考按 Q8_0 量化字节算（将 `%TEMP%` 冒烟测试正式化，补上一需求的 mul_mat_id 测试缺口）

### C. 调用链影响
- `ggml_vec_dot_gsq2_q8_0` — 调用方：`ggml-cpu.c` mul_mat/mul_mat_id one_chunk（经 traits 分发）；`ops.cpp:8442` CPU fattn（本需求起显式 assert 拒绝）；测试经 `mul_mat` 间接打内核
- `ggml_quantize_row_gsq2_act`（新增）— 调用方：`ggml-cpu.c` 两个转换循环；内部调用 `quantize_row_q8_0`（各 ISA 派发，行为不变）
- `ggml_vec_dot_gsq2_q8_0_generic` — 消费方从"标准 q8_0 y"改为"block_gsq2_act y"（签名不变，语义契约变更记录于 quants.h 注释；非 x86 fallback 同步新语义）
- `ggml_row_size(Q8_0, ne10)` 相关的 wdata 布局假设 — 仅 `ggml-cpu.c` 内部两 chunk + graph_plan，全部按 is_gsq2 分流，无外部可见
- llamafile `llamafile_sgemm` — GSQ2 输入直接跳过（返回 0 走 ggml 路径）

### E. 测试要点
- [unit / compile] `test-gsq2-vec-dot` 与全部 CPU 变体（14 个 DLL）链接无 undefined reference；`ggml_quantize_row_gsq2_act` / `ggml_gsq2_act_row_size` 符号存在于变体 DLL
- [unit / 数值-回归] 现有 5 用例**不修改**全绿（0xE4 bit 序 + mul_mat 对照 1e-4/1e-5 容差）
- [unit / 数值-mmid] 新增 mul_mat_id 用例 exit 0（多专家路由 + Q8_0 量化激活对照，相对容差 1e-3）
- [unit / 静态] fattn guard 存在：检索 `ops.cpp` fattn 入口含 GSQ2 assert
- [unit / 精度] 求和重排证据链：现有用例容差（1e-4/1e-5）覆盖 ulp 级重排差异（预研 mean 3.4e-07 / worst 3.4e-05，均为容差内）
- [unit / 缺口声明] `test-quantize-fns` 在 DL 构建不编（沿用既有声明，阈值不受本需求影响）
- [integration / 人工] 75k `-cmoe` decode ≥ 35 t/s（基线 ~30；预研 37.97）；需人工触发 `marvis-local-bench` + 人工判定
- [integration / 回归] 500 token 文本可读抽检

### G. 风险待办
- 遗漏 vec_dot 消费路径 → 静默错算。触发条件：存在第四处经 traits 调用且以标准 q8_0 量化 src1 的代码。影响：数值静默错误。缓解：本阶段已反搜（`->vec_dot` / `.vec_dot` 全量检索命中仅 ggml-cpu.c 两 chunk + ops.cpp fattn，均已处置）；verify 阶段可复核检索
- 工作区同文件叠加 q3_K CUDA 移植（另一独立任务）：ai-go 施工与 ai-verify 均须 stash 隔离、按 `ai_docs/verify-clean-tree-isolation.md` 执行符号探针，防止两任务 diff 混提
- 激活 wdata 增 41%（136→192 B/128 元素）：对 KV/权重内存可忽略，但 `graph_plan` 预留若漏改会写越界（缓解：MUL_MAT/MUL_MAT_ID 两处都已改，测试含 n=3 多 token 场景覆盖 ne11>1 路径）

### Env 执行环境提示
- 本次改动能否在编码环境内做语法/类型级自检？**能**（本机 VS2022+Clang+Ninja 全链路编译运行已验证过同类改动）
- 是否涉及需要容器内才能验证的行为？**否**（单测不依赖容器；75k 性能依赖本机 GPU + GSQ2 GGUF，属集成阶段人工项）
- 建议的验证范围（增量单测目标）：`test-gsq2-vec-dot`（含新增 mul_mat_id 用例）
- 是否预判需要集成测试：**是**（75k `-cmoe`，兑现 ~38 t/s 目标）；自动化程度：**需人工触发 + 人工判定**
