# verify 阶段的干净树隔离规范

> 来源：spec/main/no-tapd-gsq2-cpu-avx2-vnni-20260818/（2026-08-19，经人工确认写入）

`/ai-verify` 在 DL 共享构建目录上执行时，若工作区有**与本需求同文件的未提交改动**（典型：上一需求的产物尚未独立提交），必须隔离验证提交态，否则编译与测试验到的是未提交代码：

1. **stash 隔离**：`git stash push` 暂存全部未提交改动 → 在提交态（= `2-handoff-manifest.json.produced_by_commit`，先过 F5 校验）编译/测试 → `git stash pop` 恢复
2. **符号探针**：对重链的 CPU 变体 DLL（如 `bin\ggml-cpu-alderlake.dll`）执行 `llvm-nm --defined-only`，确认**不含**未提交代码引入的新增符号（本轮实例：wide 内核的 `ggml_quantize_row_gsq2_act`）——防止 ninja 增量残留导致假验证
3. **backend 有效性**：单测输出须确认实际加载的 CPU 变体含目标 ISA 特性（如 `AVX2=1`），排除 generic-vs-generic 假绿
4. verify 完成后 build 目录停留在提交态产物；恢复未提交改动后继续开发时 ninja 会自动重编，无需手工处理

来源证据：3-verify-report.md revision 2 §A（隔离说明 + 符号探针记录）。
