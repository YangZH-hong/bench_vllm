#!/usr/bin/env bash
# 多轮热路径压测一键脚本（模拟 agent 真实使用：会话累积 + prefix cache 复用）
# 与之前 run_bench.sh（单发冷路径）互补。本脚本测的是"对话越聊越长"的真实热路径。
# 用法：bash run_mt_bench.sh              # 无思考间隔，连续打，测并发上限
#       THINK="20-50" bash run_mt_bench.sh  # 带20-50s思考间隔，模拟真实节奏

set -uo pipefail

# ============ 配置 ============
CONTAINER="Coria3.6"
MODEL="Coria3.6"
TOKENIZER="/model"
URL="http://127.0.0.1:8000"

NUM_SESSIONS=100              # 每个并发档跑多少会话
CONCURRENCIES=(1 4 8 16 24)  # 并发档（最高24=服务 max-num-seqs 上限）
SEED=42
# 对数正态终止长度分布（已锁定）
MEDIAN=50000
P95=150000
LO=16000
HI=240000
PREFIX_LEN=8000

THINK="${THINK:-}"           # 环境变量传入，如 THINK="20-50"；空=无间隔
RESULT_DIR="./results_mt"
# ==============================

echo "==> 检查容器 ${CONTAINER} ..."
docker inspect -f '{{.State.Running}}' "$CONTAINER" >/dev/null 2>&1 \
  || { echo "   找不到运行中的容器 ${CONTAINER}"; exit 1; }
echo "   OK"

echo "==> 确认容器内依赖（aiohttp / numpy / transformers）..."
docker exec "$CONTAINER" bash -c 'python3 -c "import aiohttp,numpy,transformers" 2>/dev/null' \
  || { echo "   缺依赖，尝试安装 aiohttp ..."; \
       docker exec "$CONTAINER" bash -c 'pip install aiohttp --break-system-packages -q || pip install aiohttp -q'; }
echo "   OK"

echo "==> 准备工作目录并拷入客户端 ..."
docker exec "$CONTAINER" bash -c 'rm -rf /tmp/mt_work && mkdir -p /tmp/mt_work'
docker cp mt_bench.py     "${CONTAINER}:/tmp/mt_work/mt_bench.py"
docker cp summarize_mt.py "${CONTAINER}:/tmp/mt_work/summarize_mt.py"
echo "   OK"
echo

THINK_ARG=""
if [ -n "$THINK" ]; then
  THINK_ARG="--think-time $THINK"
  echo "==> 思考间隔模式：每轮间隔 ${THINK}s（模拟真实节奏，可上更高并发）"
else
  echo "==> 无思考间隔模式：连续打，测并发压力上限"
fi
echo

for c in "${CONCURRENCIES[@]}"; do
  echo "==> [并发 ${c}] 启动 ${NUM_SESSIONS} 个会话 ..."
  docker exec "$CONTAINER" bash -c "
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /tmp/mt_work
    python3 mt_bench.py \
      --url '${URL}' --model '${MODEL}' --tokenizer '${TOKENIZER}' \
      --concurrency ${c} --num-sessions ${NUM_SESSIONS} \
      --median ${MEDIAN} --p95 ${P95} --lo ${LO} --hi ${HI} \
      --prefix-len ${PREFIX_LEN} --seed ${SEED} ${THINK_ARG} \
      --result-filename result_mt_c${c}.json
  " && echo "   [并发 ${c}] 完成" || echo "   [并发 ${c}] 出错，继续下一档"
  echo
done

echo "==> 拷回结果到宿主机 ${RESULT_DIR}/ ..."
mkdir -p "$RESULT_DIR"
docker cp "${CONTAINER}:/tmp/mt_work/." "$RESULT_DIR/"
echo "   完成"
echo

echo "==> 汇总 ..."
docker exec "$CONTAINER" bash -c 'cd /tmp/mt_work && python3 summarize_mt.py'
