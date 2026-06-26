#!/usr/bin/env python3
"""多轮有状态压测客户端 —— 模拟 agent 真实使用（热路径 / prefix cache 命中）。

与 vllm bench serve 的本质区别：
  vllm bench 是【无状态单发】：每条请求独立、全新内容、全量 prefill。
  本客户端是【有状态多轮】：每个会话从短开始，逐轮累积历史，每轮请求都带完整历史，
  逐字节命中 prefix cache，只对新增部分增量 prefill —— 这才是真实 agent 的样子。

会话模型（已锁定）：
  - 开局 8K 固定 system（命中 prefix cache）
  - 每个会话采样一个【终止长度】：对数正态 中位50K / P95150K，截断[16K,240K]
  - 多轮循环：每轮 user 新增 1K-8K 随机；模型输出 200-700 随机（用 max_tokens 控制）
  - 把模型真实回复拼回历史，累积上下文达到终止长度则结束该会话
  - 关闭思考链（enable_thinking=false）

并发：asyncio 控制，--concurrency 个会话同时进行，共跑 --num-sessions 个会话。
思考间隔：默认无；--think-time "20-50" 启用，每轮之间随机等待（模拟真人阅读）。

用法（容器内，需要 /model tokenizer 和能访问的服务）：
  python3 mt_bench.py --url http://127.0.0.1:8000 --model Coria3.6 --tokenizer /model \
      --concurrency 8 --num-sessions 100 --seed 42 \
      --result-filename result_mt_c8.json
  # 加思考间隔：  --think-time 20-50
"""
import argparse, asyncio, json, sys, time, random
import numpy as np

# ---------------- 文本填充：用真实 tokenizer 精确控制 token 数 ----------------
class Filler:
    def __init__(self, tokenizer_path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        vs = self.tok.vocab_size
        self.pool = list(range(1000, min(vs, 30000)))

    def make(self, target_tok, rng):
        """生成 re-encode 后精确等于 target_tok 个 token 的文本。"""
        if target_tok <= 0:
            return ""
        ids = rng.choice(self.pool, size=int(target_tok * 1.05) + 5, replace=True).tolist()
        text = self.tok.decode(ids, skip_special_tokens=True)
        enc = self.tok.encode(text, add_special_tokens=False)
        if len(enc) >= target_tok:
            enc = enc[:target_tok]
        else:
            enc = enc + rng.choice(self.pool, size=target_tok - len(enc), replace=True).tolist()
        return self.tok.decode(enc, skip_special_tokens=True)

    def count(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))


# ---------------- 对数正态终止长度采样 ----------------
def sample_termination_lengths(n, median, p95, lo, hi, seed):
    rng = np.random.default_rng(seed)
    mu_log = np.log(median)
    sigma_log = np.log(p95 / median) / 1.645
    out = []
    while len(out) < n:
        s = rng.lognormal(mu_log, sigma_log, size=n)
        s = s[(s >= lo) & (s <= hi)]
        out.extend(s.tolist())
    return [int(x) for x in out[:n]]


# ---------------- 单个会话的多轮执行 ----------------
async def run_session(session_id, term_len, filler, args, http):
    """执行一个会话：多轮累积直到上下文达到 term_len。返回本会话的逐轮结果列表。"""
    import aiohttp
    rng = np.random.default_rng(args.seed + session_id * 7919)  # 每会话独立可复现

    local = []  # 本会话逐轮结果（并发安全：每会话独立）
    system_text = filler.shared_prefix  # 所有会话共享，命中 prefix cache
    messages = [{"role": "system", "content": system_text}]
    ctx_tokens = filler.prefix_len  # 当前累计上下文 token（近似）

    turn_idx = 0
    think_lo, think_hi = args._think_range  # None 或 (lo,hi)

    while ctx_tokens < term_len:
        turn_idx += 1
        # 本轮 user 新增 1K-8K 随机
        add = int(rng.integers(1000, 8001))
        # 不要让这一轮把上下文冲过终止线太多（最后一轮收一下）
        remaining = term_len - ctx_tokens
        if add > remaining and remaining > 200:
            add = max(200, int(remaining))
        user_text = filler.make(add, rng)
        messages.append({"role": "user", "content": user_text})
        ctx_tokens += add

        out_len = int(rng.integers(200, 701))  # 模型输出 200-700 随机

        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": out_len,
            "temperature": 0.7,
            "ignore_eos": True,  # 强制输出满 out_len，保证可比
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
        }

        # 发请求，流式读，记录 TTFT / 完成时间
        ctx_before = ctx_tokens  # 本轮请求时的上下文长度（用于分桶）
        t_start = time.perf_counter()
        ttft = None
        n_out = 0
        ok = True
        err = None
        reply = ""
        try:
            async with http.post(
                f"{args.url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=args.timeout),
            ) as resp:
                if resp.status != 200:
                    ok = False
                    err = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                else:
                    collected = []
                    async for line in resp.content:
                        if not line:
                            continue
                        s = line.decode("utf-8", "ignore").strip()
                        if not s.startswith("data:"):
                            continue
                        data = s[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except Exception:
                            continue
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        piece = delta.get("content") or ""
                        if piece:
                            if ttft is None:
                                ttft = (time.perf_counter() - t_start) * 1000.0
                            n_out += 1
                            collected.append(piece)
                    reply = "".join(collected)
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"

        t_end = time.perf_counter()
        e2el = (t_end - t_start) * 1000.0

        if not ok:
            local.append({
                "session": session_id, "turn": turn_idx, "ctx_tokens": ctx_before,
                "ok": False, "error": err, "term_len": term_len,
            })
            # 一轮失败就结束该会话（通常是超长被拒或超时）
            break

        # 把模型回复拼回历史（保持 prefix 一致性，下一轮复用缓存）
        if not reply:
            reply = filler.make(out_len, rng)  # 兜底，保证历史增长
        messages.append({"role": "assistant", "content": reply})
        ctx_tokens += n_out if n_out > 0 else out_len

        # tpot：每输出 token 平均时间
        gen_ms = e2el - (ttft if ttft is not None else e2el)
        tpot = (gen_ms / max(1, n_out - 1)) if n_out and n_out > 1 else None

        local.append({
            "session": session_id, "turn": turn_idx, "ctx_tokens": ctx_before,
            "ok": True, "ttft_ms": ttft, "e2el_ms": e2el, "tpot_ms": tpot,
            "out_tokens": n_out, "added_input": add, "term_len": term_len,
        })

        # 思考间隔
        if think_lo is not None and ctx_tokens < term_len:
            await asyncio.sleep(random.uniform(think_lo, think_hi))

    return local


async def main_async(args):
    import aiohttp

    print(f"[mt] 加载 tokenizer 并构造共享前缀 ...", file=sys.stderr)
    filler = Filler(args.tokenizer)
    filler.prefix_len = args.prefix_len
    prefix_rng = np.random.default_rng(12345)
    filler.shared_prefix = filler.make(args.prefix_len, prefix_rng)

    # 采样每个会话的终止长度
    term_lengths = sample_termination_lengths(
        args.num_sessions, args.median, args.p95, args.lo, args.hi, args.seed
    )
    arr = np.array(term_lengths)
    print(f"[mt] {args.num_sessions} 个会话终止长度: 中位={int(np.median(arr))} "
          f"均值={int(arr.mean())} p90={int(np.percentile(arr,90))} max={arr.max()}",
          file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    results = []

    # 进度状态（被会话协程更新，被进度条协程读取）
    progress = {
        "sessions_done": 0,
        "sessions_total": args.num_sessions,
        "running": 0,          # 当前在跑的会话数
        "turns_ok": 0,
        "turns_failed": 0,
        "out_tokens": 0,
    }

    def fmt_eta(done, total, elapsed):
        if done == 0:
            return "--:--"
        rate = elapsed / done
        remain = rate * (total - done)
        m, s = divmod(int(remain), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    async def progress_bar(t_start):
        bar_w = 30
        while progress["sessions_done"] < progress["sessions_total"]:
            elapsed = time.perf_counter() - t_start
            done = progress["sessions_done"]
            total = progress["sessions_total"]
            frac = done / total if total else 0
            filled = int(bar_w * frac)
            bar = "█" * filled + "░" * (bar_w - filled)
            tput = progress["out_tokens"] / elapsed if elapsed > 0 else 0
            eta = fmt_eta(done, total, elapsed)
            line = (f"\r  [{bar}] {done}/{total} 会话 | "
                    f"在跑 {progress['running']:>2} | "
                    f"成功轮 {progress['turns_ok']} 失败 {progress['turns_failed']} | "
                    f"{tput:6.0f} tok/s | 已用 {int(elapsed)}s ETA {eta}   ")
            print(line, end="", file=sys.stderr, flush=True)
            await asyncio.sleep(1.0)
        # 收尾画一次 100%
        elapsed = time.perf_counter() - t_start
        bar = "█" * bar_w
        print(f"\r  [{bar}] {progress['sessions_total']}/{progress['sessions_total']} 会话 | "
              f"成功轮 {progress['turns_ok']} 失败 {progress['turns_failed']} | "
              f"已用 {int(elapsed)}s 完成{' ':20}", file=sys.stderr, flush=True)

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as http:
        sessions = list(range(args.num_sessions))
        # 并发控制：同时最多 concurrency 个会话在跑
        sem_sessions = asyncio.Semaphore(args.concurrency)

        async def guarded(sid):
            async with sem_sessions:
                progress["running"] += 1
                local = await run_session(sid, term_lengths[sid], filler, args, http)
                results.extend(local)
                progress["turns_ok"] += sum(1 for r in local if r.get("ok"))
                progress["turns_failed"] += sum(1 for r in local if not r.get("ok"))
                progress["out_tokens"] += sum(r["out_tokens"] for r in local if r.get("ok"))
                progress["running"] -= 1
                progress["sessions_done"] += 1

        t0 = time.perf_counter()
        bar_task = asyncio.create_task(progress_bar(t0))
        await asyncio.gather(*[guarded(s) for s in sessions])
        await bar_task  # 等进度条收尾
        wall = time.perf_counter() - t0

    # 汇总
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    total_out = sum(r["out_tokens"] for r in ok)
    out = {
        "config": {
            "concurrency": args.concurrency, "num_sessions": args.num_sessions,
            "median": args.median, "p95": args.p95, "lo": args.lo, "hi": args.hi,
            "prefix_len": args.prefix_len, "think_time": args.think_time,
            "seed": args.seed,
        },
        "wall_time_s": wall,
        "num_turns_ok": len(ok),
        "num_turns_failed": len(bad),
        "total_output_tokens": total_out,
        "output_throughput_tok_s": total_out / wall if wall > 0 else 0,
        "turn_throughput_per_s": len(ok) / wall if wall > 0 else 0,
        "term_lengths": term_lengths,
        "turns": results,  # 逐轮明细，供 summarize 分桶
    }
    with open(args.result_filename, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[mt] 完成: {len(ok)} 轮成功, {len(bad)} 轮失败, "
          f"墙钟 {wall:.1f}s, 输出吞吐 {out['output_throughput_tok_s']:.1f} tok/s "
          f"-> {args.result_filename}", file=sys.stderr)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--num-sessions", type=int, default=100)
    # 对数正态终止长度（已锁定默认）
    ap.add_argument("--median", type=int, default=50000)
    ap.add_argument("--p95", type=int, default=150000)
    ap.add_argument("--lo", type=int, default=16000)
    ap.add_argument("--hi", type=int, default=240000)
    ap.add_argument("--prefix-len", type=int, default=8000)
    ap.add_argument("--think-time", default=None,
                    help='思考间隔秒数范围，如 "20-50"；不传则无间隔（连续打，测并发上限）')
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--result-filename", default="result_mt.json")
    args = ap.parse_args()
    # 解析 think-time
    if args.think_time:
        lo, hi = args.think_time.split("-")
        args._think_range = (float(lo), float(hi))
    else:
        args._think_range = (None, None)
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    asyncio.run(main_async(args))
