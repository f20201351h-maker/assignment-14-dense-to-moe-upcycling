#!/usr/bin/env bash
# Exact command sequence used for this experiment (Modal CLI 1.6.0, profile WORKSPACE).
# Each step was launched by hand after checking the previous one; this file records them.
set -euo pipefail
export PYTHONUTF8=1 PYTHONWARNINGS=ignore MSYS_NO_PATHCONV=1  # last one: stop Git Bash on Windows rewriting /vol paths
LR=${LR:-1e-3}   # chosen from the LR smoke test below (see README)

# 0. local correctness checks on a random-init model at the real config (CPU, float64)
python src/checks.py --out results/checks_random_init.txt

# 1. data: byte-level BPE 8192 on the TinyStories train split, tokenize train + validation (Modal CPU)
modal run modal_app.py::prep_data

# 2. two-minute throughput benchmark, dense vs MoE, A10G and L40S
modal run modal_app.py::bench

# 3. LR smoke test: 600 dense steps per LR on the same schedule shape as the full run
for lr in 6e-4 1e-3 2e-3; do
  modal run --detach modal_app.py::train --args "--stage 1 --data_dir /vol/data --run_dir /vol/runs/lrsweep_$lr --peak_lr $lr --s1_steps 600 --s2_steps 11600 --ckpt_every 600" &
done
wait

# 3b. stage-2 smoke test on GPU: convert the 600-step lr=1e-3 checkpoint, 400 + 1000 steps per arm
#     (logs/smoke_stage2/; this is where bias-only balancing left dead experts in layers 2 and 7)
SM="--stage 2 --data_dir /vol/data --stage1_ckpt /vol/runs/lrsweep_1e-3/stage1_final.pt --peak_lr 1e-3 --s1_steps 600 --s2_steps 11600"
for arm in moe dense; do
  modal run --detach modal_app.py::train --args "$SM --arm $arm --run_dir /vol/runs/smoke_$arm --max_steps 400"
  modal run --detach modal_app.py::train --args "$SM --arm $arm --run_dir /vol/runs/smoke_$arm --max_steps 1000"  # resumes at step 1000
done

# 4. stage 1: dense, 6100 steps x 32768 tokens
modal run --detach modal_app.py::train --args "--stage 1 --data_dir /vol/data --run_dir /vol/runs/stage1 --peak_lr $LR"

# 5. correctness checks on the real stage-1 checkpoint (local CPU, float64)
modal volume get s14-moe-upcycle /runs/stage1/stage1_final.pt ckpt/stage1_final.pt
python src/checks.py --ckpt ckpt/stage1_final.pt --out results/checks_stage1_checkpoint.txt

# 6. stage 2: three arms from the same checkpoint, in parallel (dense control; MoE as planned; MoE + aux fallback)
S2="--stage 2 --data_dir /vol/data --stage1_ckpt /vol/runs/stage1/stage1_final.pt --peak_lr $LR"
modal run --detach modal_app.py::train --args "$S2 --arm dense --run_dir /vol/runs/stage2_dense" &
modal run --detach modal_app.py::train --args "$S2 --arm moe --run_dir /vol/runs/stage2_moe" &
modal run --detach modal_app.py::train --args "$S2 --arm moe --aux_coef 0.01 --run_dir /vol/runs/stage2_moe_aux" &
wait

# 7. fetch logs and build figures / headline numbers
for r in stage1:stage1_dense stage2_dense:stage2_dense stage2_moe:stage2_moe stage2_moe_aux:stage2_moe_aux; do
  modal volume get --force s14-moe-upcycle /runs/${r%%:*}/log.jsonl logs/${r##*:}.jsonl
done
python src/analyze.py --logs logs --out figures --results results

# 8. data checks added after training: exact train/validation story overlap, and re-evaluation of the key
#    checkpoints on validation stories that never occur in train ("clean") vs ones that do ("dup")
modal run modal_app.py::overlap      # -> results/train_val_overlap.json
modal run modal_app.py::clean_eval   # -> results/eval_by_train_overlap.json

# 9. cost of this app (per-app numbers; the workspace total includes other apps)
modal billing report --for today --show-resources --json > modal_billing_report_today.json  # raw account report, summarised in results/modal_cost.json (raw file not committed)
