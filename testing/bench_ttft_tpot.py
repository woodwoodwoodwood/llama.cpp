#!/usr/bin/env python3
"""
测试 llama-server 在不同输入长度下的 TTFT / TPOT / decode t/s。

通过 llama-server 的 /v1/completions 端点（非流式），从返回的 `timings` 字段
直接读取 prompt_ms (= TTFT) 和 predicted_ms / predicted_n (= TPOT)，精确、无需反推。

用法见: python3 bench_ttft_tpot.py --help

前置条件:
  1. llama-server 已启动，且 /v1/completions 可访问（支持 token id 数组 prompt）。
     推荐启动参数（按需调整 -c / GPU）:
       CUDA_VISIBLE_DEVICES=6 ./build/bin/llama-server \
         -m /data1/models/qwen36-35b-a3b-gsq2.gguf \
         -c 130000 -t 8 -ngl 99 --port 8901 --host 0.0.0.0 \
         -ctk q8_0 -ctv q8_0 -fa on -np 1
  2. transformers + 一个含 tokenizer.json 的目录（与被测模型同系列的 tokenizer 均可，
     量化不改变 tokenizer）。

原理:
  - 用 HF tokenizer 把真实长文本编码成 token id 序列，取前 n 个。
  - 直接把 token id 数组作为 prompt 发给 server（server 不重新 tokenize，
    保证 actual_pp 严格等于目标长度，不会膨胀）。
  - 读返回的 timings：prompt_ms / prompt_n / predicted_ms / predicted_n。
"""
import argparse
import json
import os
import sys
import time
import urllib.request

# 默认的真实长文本来源（拼起来够 100k+ token）。可被 --text-file 覆盖。
DEFAULT_TEXT_FILES = [
    "/usr/local/app/llama.cpp/README.md",
    "/usr/local/app/llama.cpp/docs/server.md",
    "/usr/local/app/llama.cpp/AGENTS.md",
    "/usr/local/app/llama.cpp/docs/build.md",
]

# 本地 WSL 副本上的 fallback 文本（如果默认文件不存在）
LOCAL_FALLBACK_FILES = [
    "README.md",
    "docs/server.md",
    "AGENTS.md",
]


def build_big_text(text_files, repeat=30):
    """读取多个文本文件拼成大文本。"""
    texts = []
    for f in text_files:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                texts.append(fh.read())
    if not texts:
        raise FileNotFoundError(
            f"找不到任何文本文件: {text_files}。请用 --text-file 指定。"
        )
    return ("\n\n".join(texts)) * repeat


def make_opener(no_proxy=True):
    """构建一个不走代理的 opener（避免 localhost 被 Squid 等代理拦截返回 403）。"""
    if no_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def parse_args():
    p = argparse.ArgumentParser(
        description="测试 llama-server 不同输入长度下的 TTFT/TPOT/decode t/s",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8901",
                   help="llama-server 地址（不含 /v1/completions）")
    p.add_argument("--tokenizer", default="/data1/models/Qwen3.6-35B-A3B",
                   help="HF tokenizer 目录（含 tokenizer.json）。与被测模型同系列即可")
    p.add_argument("--lens", default="8192,16384,25000,30000,35000,45000,75000,100000",
                   help="要测试的输入长度（token 数），逗号分隔")
    p.add_argument("--gen", type=int, default=512, help="每个请求的生成 token 数")
    p.add_argument("--text-file", action="append", default=None,
                   help="用于凑长度的文本文件（可多次指定）。默认用 llama.cpp 文档")
    p.add_argument("--text-repeat", type=int, default=30,
                   help="文本重复次数（保证总 token 数够最大输入长度）")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=600, help="单请求超时（秒）")
    p.add_argument("--cache-prompt", action="store_true",
                   help="开启 server 端 prompt cache（默认关闭，保证每次真实 prefill）")
    p.add_argument("--no-prefix-reuse", action="store_true",
                   help="每个长度从文本的不同位置取样，避免跨请求前缀复用 "
                        "（消除 LCP cache 复用，让 server 端 n_tokens 也精确等于目标，TTFT 最严格）")
    p.add_argument("--output", default=None,
                   help="把结果存成 JSON 文件（可选）")
    p.add_argument("--use-proxy", action="store_true",
                   help="走系统代理（默认不走，避免 localhost 被拦）")
    return p.parse_args()


def main():
    args = parse_args()
    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    opener = make_opener(no_proxy=not args.use_proxy)

    # 准备文本 + tokenizer
    text_files = args.text_file if args.text_file else (DEFAULT_TEXT_FILES + LOCAL_FALLBACK_FILES)
    big = build_big_text(text_files, repeat=args.text_repeat)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: 需要 transformers，请 pip install transformers", file=sys.stderr)
        sys.exit(1)

    print(f"加载 tokenizer: {args.tokenizer}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    all_ids = tok.encode(big, add_special_tokens=False)
    print(f"文本总 token 数: {len(all_ids)}", file=sys.stderr)
    print(file=sys.stderr)

    # 表头
    print(f"{'target':>7} {'actual_pp':>9} | {'TTFT(ms)':>9} {'TPOT(ms)':>9} "
          f"{'decode_t/s':>10} {'pp_t/s':>9} {'gen_n':>5} {'e2e(ms)':>9}")
    print("-" * 90)

    results = []
    import random
    rng = random.Random(12345)  # 固定种子，可复现
    for idx, n in enumerate(lens):
        if n > len(all_ids):
            print(f"{n:>7}         - |  文本不够（{len(all_ids)} tokens），跳过")
            continue

        if args.no_prefix_reuse:
            # 每个长度从不同位置取样，避免与之前请求有共同前缀，
            # 从而消除 server 端 LCP cache 复用 -> n_tokens 严格等于目标，TTFT 最严格。
            # 用 idx 做种子保证不同长度取不同段，但每次运行取相同段（可复现）。
            max_start = len(all_ids) - n - 100
            if max_start <= 0:
                print(f"{n:>7}         - |  文本不够做无前缀复用取样，跳过")
                continue
            start = (idx * (len(all_ids) // (len(lens) + 1))) % max_start
            prompt_ids = all_ids[start:start + n]
        else:
            # 默认：都从文本开头取前 n 个（递增长度会有共同前缀，server 可能复用前缀 KV）
            prompt_ids = all_ids[:n]

        body = json.dumps({
            "model": "x",
            "prompt": prompt_ids,            # 直接发 token id 数组，server 不重新 tokenize
            "max_tokens": args.gen,
            "temperature": args.temperature,
            "stream": False,
            "cache_prompt": args.cache_prompt,
        }).encode()
        req = urllib.request.Request(
            f"{args.base_url}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            t0 = time.time()
            r = opener.open(req, timeout=args.timeout)
            wall = (time.time() - t0) * 1000
            d = json.loads(r.read())
            t = d["timings"]
            ttft = t["prompt_ms"]
            pred_ms = t["predicted_ms"]
            pred_n = t["predicted_n"]
            actual_pp = t["prompt_n"]

            if pred_n == 0:
                tpot = 0.0
                decode_ts = 0.0
                flag = " <0生成"
            elif pred_n < 10:
                tpot = pred_ms / pred_n
                decode_ts = pred_n / (pred_ms / 1000) if pred_ms > 0 else 0.0
                flag = f" <只生成{pred_n}"
            else:
                tpot = pred_ms / pred_n
                decode_ts = pred_n / (pred_ms / 1000) if pred_ms > 0 else 0.0
                flag = ""

            # 检查 actual_pp 是否精确等于目标（不一致说明有前缀复用/token 过滤）
            if actual_pp != n:
                flag += f" <pp偏差{actual_pp - n:+d}>"

            pp_ts = actual_pp / (ttft / 1000) if ttft > 0 else 0.0

            print(f"{n:>7} {actual_pp:>9} | {ttft:>9.1f} {tpot:>9.2f} "
                  f"{decode_ts:>10.1f} {pp_ts:>9.1f} {pred_n:>5} {wall:>9.0f}{flag}")

            results.append({
                "target_tokens": n,
                "actual_pp": actual_pp,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "decode_tps": decode_ts,
                "prefill_tps": pp_ts,
                "gen_n": pred_n,
                "e2e_ms": wall,
                "flag": flag.strip() if flag else None,
            })
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", errors="ignore")[:200]
            print(f"{n:>7}         - |  FAIL: HTTP {e.code} {body_txt}")
            results.append({"target_tokens": n, "error": f"HTTP {e.code}: {body_txt}"})
        except Exception as e:
            print(f"{n:>7}         - |  FAIL: {type(e).__name__} {e}")
            results.append({"target_tokens": n, "error": f"{type(e).__name__}: {e}"})

    # 存 JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "base_url": args.base_url,
                "tokenizer": args.tokenizer,
                "gen": args.gen,
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已存: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
