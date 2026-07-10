#!/usr/bin/env bash
# Claude: auto-resume wrapper for the 4B Jacobian fit (U4 step 3).
#
# The 3090 is shared with the user's live VS Code Jupyter kernel, whose bursty
# GPU/RAM load repeatedly OOM-killed this long-lived fit:
#   551 host-OOM  (SIGKILL, no traceback) at n_done=36  (kernel ~1.5GB)
#   552 CUDA-OOM  (traceback, exit 1)     at n_done=45  (kernel grew to 8.18GB)
#   553 host-OOM  (SIGKILL, no traceback) at n_done=64
# jlens checkpoints every prompt and resumes, so the accumulated Jacobian is
# never lost -- each kill only costs one model reload. So instead of predicting
# the user's bursts, just relaunch until the fit exits 0 (success writes
# artifacts/u4_loopclose.txt). expandable_segments defrags VRAM; dim_batch=4
# (in the .py) keeps this a polite ~10.5GB co-tenant.
#
# MAX_RETRIES caps runaway: with ~15-20 prompts/run and 336 remaining, an
# all-bursty night needs <25 relaunches; 40 is safe headroom. If n_done stops
# advancing across retries (logged below), that's a real bug, not an OOM --
# stop and debug rather than spin.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=artifacts/qwen3-4b-authority.ckpt
MAX_RETRIES=40
n=0
ndone() { .venv/bin/python -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False)['n_done'])" 2>/dev/null; }
until .venv/bin/python scripts/u4_step3_fit4b.py; do
  n=$((n + 1))
  echo "[retry-wrapper] fit exited non-zero (OOM?); attempt $n/$MAX_RETRIES, n_done=$(ndone), resuming in 90s"
  if [ "$n" -ge "$MAX_RETRIES" ]; then
    echo "[retry-wrapper] hit MAX_RETRIES=$MAX_RETRIES at n_done=$(ndone) -- likely a real bug, not OOM. Stopping."
    exit 1
  fi
  sleep 90
done
echo "[retry-wrapper] fit completed after $n retries, n_done=$(ndone)"
