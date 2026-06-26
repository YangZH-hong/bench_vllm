#!/usr/bin/env python3
"""汇总多轮压测结果 result_mt_c*.json。
用法: python3 summarize_mt.py
核心看两件事：
  1) 并发档对比：吞吐、整体延迟
  2) 热路径核心 —— TTFT 随【当前上下文长度】怎么爬升（增量 prefill 的代价）
"""
import json, glob, re, sys
import numpy as np

files = glob.glob("result_mt_c*.json")
if not files:
    sys.exit("没找到 result_mt_c*.json，先跑 run_mt_bench.sh")

def cnum(p):
    m = re.search(r"_c(\d+)\.json$", p)
    return int(m.group(1)) if m else 0

files = sorted(files, key=cnum)
runs = [(cnum(p), json.load(open(p))) for p in files]

def pct(arr, p):
    return np.percentile(arr, p) if len(arr) else float("nan")

# ---------- 第一部分：并发档对比 ----------
print("\n========= 多轮热路径压测 · 并发档对比 =========")
print("分布: 对数正态 中位50K/P95150K 截断[16K,240K]，每轮输入+1~8K，输出200~700，关思考链")
hdr = ["并发", "成功轮", "失败", "墙钟s", "输出tok/s", "轮/s",
       "TTFT_p50", "TTFT_p95", "TTFT_p99", "E2EL_p50", "TPOT_p50"]
print("  ".join(f"{h:>9}" for h in hdr))
print("-" * (11 * len(hdr)))
for c, d in runs:
    ok = [t for t in d["turns"] if t.get("ok")]
    ttfts = [t["ttft_ms"] for t in ok if t.get("ttft_ms") is not None]
    e2els = [t["e2el_ms"] for t in ok if t.get("e2el_ms") is not None]
    tpots = [t["tpot_ms"] for t in ok if t.get("tpot_ms") is not None]
    line = [
        str(c), str(d["num_turns_ok"]), str(d["num_turns_failed"]),
        f"{d['wall_time_s']:.0f}", f"{d['output_throughput_tok_s']:.1f}",
        f"{d['turn_throughput_per_s']:.2f}",
        f"{pct(ttfts,50):.0f}", f"{pct(ttfts,95):.0f}", f"{pct(ttfts,99):.0f}",
        f"{pct(e2els,50):.0f}", f"{pct(tpots,50):.0f}",
    ]
    print("  ".join(f"{x:>9}" for x in line))

# ---------- 第二部分：TTFT 随上下文长度爬升（每个并发档） ----------
# 这是热路径的核心：上下文越长，增量 prefill 越重，TTFT 应逐桶上升
CTX_BUCKETS = [(0,32000),(32000,64000),(64000,96000),(96000,128000),
               (128000,160000),(160000,200000),(200000,9999999)]
def bname(lo,hi):
    h = "240K+" if hi>900000 else f"{hi//1000}K"
    return f"{lo//1000}-{h}"

print("\n========= TTFT 随【请求时上下文长度】爬升（热路径核心） =========")
print("（每轮请求按发起时的上下文长度分桶；TTFT 上升即增量 prefill 变重）")
for c, d in runs:
    ok = [t for t in d["turns"] if t.get("ok") and t.get("ttft_ms") is not None]
    if not ok:
        print(f"\n[并发 {c}] 无成功轮，跳过")
        continue
    print(f"\n[并发 {c}]  （成功轮 {len(ok)}）")
    print(f"  {'上下文桶':>12} {'轮数':>5} {'TTFT_p50':>10} {'TTFT_p95':>10} {'E2EL_p50':>10} {'TPOT_p50':>10}")
    for lo,hi in CTX_BUCKETS:
        sub = [t for t in ok if lo <= t["ctx_tokens"] < hi]
        if not sub:
            continue
        tt = [t["ttft_ms"] for t in sub]
        ee = [t["e2el_ms"] for t in sub]
        tp = [t["tpot_ms"] for t in sub if t.get("tpot_ms") is not None]
        print(f"  {bname(lo,hi):>12} {len(sub):>5} "
              f"{pct(tt,50):>10.0f} {pct(tt,95):>10.0f} {pct(ee,50):>10.0f} "
              f"{(pct(tp,50) if tp else float('nan')):>10.0f}")

# ---------- 第三部分：会话长度分布与轮数 ----------
print("\n========= 会话画像 =========")
for c, d in runs:
    tl = np.array(d["term_lengths"])
    ok = [t for t in d["turns"] if t.get("ok")]
    # 每会话轮数
    import collections
    turns_per = collections.Counter(t["session"] for t in ok)
    tp = list(turns_per.values())
    print(f"  [并发 {c}] 会话数={len(tl)}  终止长度 中位={int(np.median(tl))//1000}K "
          f"p90={int(np.percentile(tl,90))//1000}K max={tl.max()//1000}K  "
          f"每会话轮数 中位={int(np.median(tp)) if tp else 0} max={max(tp) if tp else 0}")

# 失败详情
print("\n— 失败轮详情（前若干条）—")
any_fail = False
for c, d in runs:
    bad = [t for t in d["turns"] if not t.get("ok")]
    for t in bad[:3]:
        any_fail = True
        print(f"  [并发 {c}] 会话{t['session']} 轮{t['turn']} "
              f"ctx≈{t['ctx_tokens']//1000}K: {t.get('error','?')[:120]}")
if not any_fail:
    print("  无失败轮")

print("\n提示: 对比'多轮热路径'与之前'单发冷路径'的同长度 TTFT——")
print("      热路径因 prefix cache 命中，长上下文 TTFT 应显著低于冷路径全量 prefill。")
