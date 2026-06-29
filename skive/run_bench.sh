#!/usr/bin/env bash
# Stage 6 sweep: FullKV vs eviction at matched KV budget + workload.
# Eviction is V1-runner only -> force V1 for every run.
set -u
export VLLM_USE_V2_MODEL_RUNNER=0
cd "$(dirname "$0")"

COMMON=(--blocks 1500 --nreqs 96 --prompt-len 200 --gen-len 200 --max-model-len 1024)

echo "### FullKV ###"
python benchmark.py "${COMMON[@]}"
echo "### evict budget=20 (~80%) ###"
python benchmark.py "${COMMON[@]}" --evict --budget 20 --sink 2 --local 4
echo "### evict budget=13 (~50%) ###"
python benchmark.py "${COMMON[@]}" --evict --budget 13 --sink 2 --local 4
echo "### evict budget=8 (~32%) ###"
python benchmark.py "${COMMON[@]}" --evict --budget 8 --sink 2 --local 4
echo "BENCH_DONE"
