#!/usr/bin/env python3
"""
分析 nsys 报告，把 CUDA kernel 按 prefill / decode 阶段分开统计开销。

用法:
  先导出 trace CSV:
    nsys stats --report gputrace_stats --format csv -o gsq2_trace gsq2_profile.nsys-rep
  再跑本脚本:
    python3 bench_nsys_phase.py gsq2_trace_gputrace_stats.csv --prefill-ms 2086 --label GSQ2
    python3 bench_nsys_phase.py q2k_trace_gputrace_stats.csv  --prefill-ms 2742 --label Q2_K-M

原理:
  llama-bench -p 8192 -n 512 先 prefill 再 decode。从 nsys trace 里：
  - prefill 阶段: 大量 mul_mat_q (MMQ) 密集调用
  - decode 阶段: mul_mat_vec_q (MMVQ) 逐 token 调用, 频率低
  用一个分界时间点把 kernel 分到两个阶段。分界点可用：
    1) --prefill-ms 手动给（= n_prompt / pp_tps * 1000）
    2) 自动检测 MMQ 密集区间的结束（默认）

  然后按 kernel 类别（attention / MoE-FFN / 其他）在两阶段分别汇总。
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

# ---------- kernel 分类规则 ----------
def classify(name: str) -> str:
    n = name.lower()
    # attention / 序列建模
    if "gated_delta_net" in n or "ssm_conv" in n or "fwht" in n:
        return "attention"
    if "flash_attn" in n:
        return "attention"
    if "rope" in n:
        return "attention"
    if "rms_norm" in n or "l2_norm" in n:
        return "attention"
    if "soft_max" in n:
        return "attention"
    # MoE / FFN 专家
    if "mul_mat_q" in n or "mul_mat_vec_q" in n:
        return "moe_ffn"
    if "topk_moe" in n or "mm_ids_helper" in n:
        return "moe_ffn"
    if "get_rows" in n:
        return "moe_ffn"
    if "concat" in n:
        return "moe_ffn"
    if "unary_gated_op" in n or "unary_op_kernel" in n:
        return "moe_ffn"  # silu/sigmoid/softplus 属于 FFN 激活
    if "k_bin_bcast" in n:
        return "moe_ffn"  # gate 分数乘/add
    if "quantize_mmq" in n or "set_rows_quant" in n:
        return "moe_ffn"  # MMQ 辅助
    if "nvjet" in n or "xmma" in n or "gemv" in n or "gemm" in n:
        return "moe_ffn"  # cublas 矩阵乘（BF16 层），归到 MoE/FFN
    # 通用/转换
    if "convert_unary" in n or "quantize_q8" in n or "cpy" in n:
        return "other"
    if "scale_f32" in n or "op_clamp" in n or "reduce_rows" in n:
        return "other"
    return "other"


def short_name(name: str) -> str:
    """简化 kernel 名，归类用。提取核心名字 + 类型（如 mul_mat_q<42>）"""
    # 提取 mul_mat_q<(ggml_type)N> 里的 N
    m = re.search(r"mul_mat_q<\(ggml_type\)(\d+)", name)
    if m:
        return f"mul_mat_q<type{m.group(1)}>"
    m = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+)", name)
    if m:
        return f"mul_mat_vec_q<type{m.group(1)}>"
    # 其他取第一个 < 之前
    return name.split("<")[0].split("(")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="nsys gputrace_stats CSV 文件")
    ap.add_argument("--prefill-ms", type=float, default=None,
                    help="prefill 持续时间(ms)，用于手动设定分界点。不给则自动检测")
    ap.add_argument("--label", default="", help="模型标签")
    ap.add_argument("--auto-split", action="store_true",
                    help="自动检测 prefill/decode 分界（MMQ 密集区结束点）")
    args = ap.parse_args()

    # 读取 CSV：nsys gputrace_stats 的列通常是
    # Time(%),Total Time(ns),Instances,Avg(ns),Med(ns),Min(ns),Max(ns),StdDev(ns),Name  (这是汇总，不是逐条)
    # 但我们要逐条 trace。nsys 的 gputrace_stats 实际是逐条 GPU kernel 事件。
    rows = []
    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # 找到关键列：开始时间、持续时间、名字
        # nsys gputrace_stats 列名: Time,Duration,CorrId,GrdX,GrdY,GrdZ,WgrdX,WgrdY,WgrdZ,Regs,Stc,Smem,Dmem,Size,Throughput,Device,Ctx,Strm,Name
        ti = header.index("Time") if "Time" in header else 0
        di = header.index("Duration") if "Duration" in header else 1
        ni = header.index("Name") if "Name" in header else len(header) - 1
        for r in reader:
            if len(r) <= max(ti, di, ni):
                continue
            try:
                t = float(r[ti])      # 开始时间 (ns)
                d = float(r[di])      # 持续时间 (ns)
            except ValueError:
                continue
            rows.append((t, d, r[ni]))

    if not rows:
        print(f"[{args.label}] 未读到 kernel 事件，CSV 列: {header}", file=sys.stderr)
        sys.exit(1)

    rows.sort()
    t0 = rows[0][0]
    # 相对开始时间（秒）
    times = [(t - t0) / 1e9 for t, d, _ in rows]

    # ---------- 确定分界点 ----------
    if args.prefill_ms is not None:
        split_s = args.prefill_ms / 1000.0
    else:
        # 自动检测：找 mul_mat_q (MMQ) 最后一次出现的时间作为 prefill 结束
        mmq_times = [(t - t0) / 1e9 for t, d, n in rows if "mul_mat_q<" in n and "fixup" not in n]
        if mmq_times:
            split_s = max(mmq_times) + 0.01  # MMQ 结束后略加余量
        else:
            split_s = (times[-1]) / 2  # fallback 中点

    # ---------- 按阶段 + 类别汇总 ----------
    phase_cat = {"prefill": defaultdict(float), "decode": defaultdict(float)}
    phase_kernel = {"prefill": defaultdict(float), "decode": defaultdict(float)}
    phase_cat_count = {"prefill": defaultdict(int), "decode": defaultdict(int)}

    for t, d, name in rows:
        rel = (t - t0) / 1e9
        phase = "prefill" if rel < split_s else "decode"
        cat = classify(name)
        sn = short_name(name)
        phase_cat[phase][cat] += d
        phase_kernel[phase][sn] += d
        phase_cat_count[phase][cat] += 1

    # ---------- 打印 ----------
    label = args.label or args.csv
    print(f"\n{'='*70}")
    print(f"  {label}   (prefill/decode 分界点 = {split_s:.3f}s)")
    print(f"{'='*70}")

    for phase in ["prefill", "decode"]:
        total = sum(phase_cat[phase].values())
        print(f"\n[{phase.upper()}] 总 kernel 时间 = {total/1e6:.1f} ms")
        print(f"  {'类别':<14} {'时间(ms)':>12} {'占比':>8} {'调用数':>8}")
        print(f"  {'-'*46}")
        for cat in ["attention", "moe_ffn", "other"]:
            v = phase_cat[phase][cat]
            if total > 0:
                print(f"  {cat:<14} {v/1e6:>12.1f} {v/total*100:>7.1f}% {phase_cat_count[phase][cat]:>8}")

    # ---------- MoE/FFN 内细分（量化类型）----------
    print(f"\n[{label}] MoE/FFN 各 kernel 在两阶段的时间(ms):")
    print(f"  {'kernel':<28} {'prefill(ms)':>12} {'decode(ms)':>12}")
    print(f"  {'-'*54}")
    all_kernels = set(phase_kernel["prefill"]) | set(phase_kernel["decode"])
    for k in sorted(all_kernels):
        if "mul_mat" in k or "mm_ids" in k or "topk" in k or "concat" in k or "get_rows" in k \
           or "unary_gated" in k or "bin_bcast" in k or "nvjet" in k or "xmma" in k:
            pf = phase_kernel["prefill"].get(k, 0) / 1e6
            dc = phase_kernel["decode"].get(k, 0) / 1e6
            if pf > 0.05 or dc > 0.05:  # 只显示有意义的
                print(f"  {k:<28} {pf:>12.1f} {dc:>12.1f}")

    # attention 内细分
    print(f"\n[{label}] attention 各 kernel 在两阶段的时间(ms):")
    print(f"  {'kernel':<28} {'prefill(ms)':>12} {'decode(ms)':>12}")
    print(f"  {'-'*54}")
    for k in sorted(all_kernels):
        if any(s in k for s in ["gated_delta", "ssm_conv", "fwht", "flash_attn", "rope", "rms_norm", "l2_norm", "soft_max"]):
            pf = phase_kernel["prefill"].get(k, 0) / 1e6
            dc = phase_kernel["decode"].get(k, 0) / 1e6
            if pf > 0.05 or dc > 0.05:
                print(f"  {k:<28} {pf:>12.1f} {dc:>12.1f}")


if __name__ == "__main__":
    main()
