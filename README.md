# 多轮热路径压测工具 (MT-Bench v2.0)

模拟 **真实 agent 使用场景** 的多轮有状态压测客户端，专门测量 **prefix cache 命中下的增量 prefill 性能**。

---

## 目的

传统压测工具（如 `vllm bench serve`）是 **无状态单发**：每条请求都是全新内容、独立发出、走全量 prefill 冷路径。这与真实业务对不上——真实的 agent / 对话应用是 **越聊越长**，每一轮都带着完整历史发请求。

本工具测的就是这条 **热路径**：

| | `vllm bench serve`（冷路径） | 本工具（热路径） |
|---|---|---|
| 状态 | 无状态单发 | 有状态多轮累积 |
| 请求内容 | 每条全新 | 每轮带完整历史 |
| Prefill | 每次全量 prefill | 逐字节命中 prefix cache，只增量 prefill 新增部分 |
| 贴近场景 | 批处理 / 一问一答 | Agent、长对话、Coding 助手 |

**核心要回答的问题**：随着上下文越来越长，TTFT（首 token 延迟）会怎样爬升？prefix cache 到底省了多少？服务在多少并发下还扛得住？

---

## 设计思想（会话模型）

每个会话模拟一次真实的多轮对话，规则已锁定：

1. **共享 8K system 前缀**：所有会话用同一段固定 system prompt，必然命中 prefix cache。
2. **对数正态终止长度**：每个会话采样一个"聊到多长就结束"的目标长度——中位 50K / P95 150K，截断到 `[16K, 240K]`，逼近真实长尾分布。
3. **多轮累积**：
   - 每轮 user 新增 **1K–8K** 随机 token；
   - 模型输出 **200–700** 随机 token（`max_tokens` 控制，`ignore_eos` 保证输出满，结果可比）；
   - 把模型真实回复 **拼回历史**，保持 prefix 一致性，下一轮继续复用缓存；
   - 上下文累积到终止长度则结束该会话。
4. **关闭思考链**（`enable_thinking=false`），避免思考 token 干扰测量。
5. **可选思考间隔**：`--think-time "20-50"` 模拟真人阅读停顿（每轮间随机等待）；不传则连续打，测并发压力上限。

文本填充用 **真实 tokenizer** 精确控制 token 数（生成 → 解码 → 重编码裁剪），保证每轮输入长度准确。

并发由 `asyncio` 信号量控制：最多 `--concurrency` 个会话同时进行，共跑 `--num-sessions` 个会话。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `mt_bench.py` | 压测客户端核心。跑多轮会话，流式记录 TTFT / E2EL / TPOT，输出逐轮明细 JSON。 |
| `run_mt_bench.sh` | 一键脚本。检查容器、装依赖、拷脚本进容器、按多个并发档依次压测、拷回结果、调用汇总。 |
| `summarize_mt.py` | 结果汇总。读取 `result_mt_c*.json`，输出并发档对比 + TTFT 随上下文长度的爬升曲线 + 会话画像。 |

---

## 如何使用

### 方式一：一键脚本（推荐，针对 Docker 容器内服务）

前提：有一个运行中的容器（默认名 `Coria3.6`），容器内已起好 OpenAI 兼容服务（默认 `http://127.0.0.1:8000`），并有 tokenizer（默认 `/model`）。

```bash
# 无思考间隔：连续打，测并发压力上限
bash run_mt_bench.sh

# 带思考间隔：每轮间隔 20–50s，模拟真实节奏
THINK="20-50" bash run_mt_bench.sh
```

脚本会自动按 `CONCURRENCIES=(1 4 8 16 24)` 几个并发档依次跑，结果落到宿主机 `./results_mt/`，最后打印汇总。

> 改容器名 / 模型 / tokenizer / 并发档，直接编辑 `run_mt_bench.sh` 顶部的 **配置** 区。

### 方式二：直接跑客户端

```bash
python3 mt_bench.py \
    --url http://127.0.0.1:8000 \
    --model Coria3.6 \
    --tokenizer /model \
    --concurrency 8 \
    --num-sessions 100 \
    --seed 42 \
    --result-filename result_mt_c8.json

# 加思考间隔：再加 --think-time 20-50
```

依赖：`aiohttp`、`numpy`、`transformers`（容器内若缺，一键脚本会尝试自动安装 `aiohttp`）。

### 主要参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--url` | `http://127.0.0.1:8000` | 服务地址（OpenAI 兼容） |
| `--model` | *(必填)* | 模型名 |
| `--tokenizer` | *(必填)* | tokenizer 路径，用于精确控制 token 数 |
| `--concurrency` | 8 | 同时进行的会话数 |
| `--num-sessions` | 100 | 总会话数 |
| `--median` / `--p95` | 50000 / 150000 | 终止长度对数正态的中位 / P95 |
| `--lo` / `--hi` | 16000 / 240000 | 终止长度截断区间 |
| `--prefix-len` | 8000 | 共享 system 前缀长度 |
| `--think-time` | 无 | 思考间隔秒数范围，如 `20-50` |
| `--timeout` | 600 | 单请求超时（秒） |
| `--seed` | 42 | 随机种子（可复现） |
| `--result-filename` | `result_mt.json` | 结果输出文件 |

### 看结果

```bash
python3 summarize_mt.py   # 自动读取当前目录下所有 result_mt_c*.json
```

输出三部分：

1. **并发档对比**：吞吐（tok/s、轮/s）、TTFT/E2EL/TPOT 的 p50/p95/p99。
2. **TTFT 随上下文长度爬升**（热路径核心）：按"请求发起时的上下文长度"分桶，看 TTFT 怎么随上下文变长而上升——这就是增量 prefill 的代价。
3. **会话画像**：终止长度分布、每会话轮数、失败详情。

> 关键对比：同样的上下文长度下，**热路径**因 prefix cache 命中，TTFT 应显著低于 `vllm bench` 的冷路径全量 prefill。
