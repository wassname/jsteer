#!/usr/bin/env bash
# Claude: external requeue guard for the U4 4B Jacobian fit.
#
# WHY not just the in-task retry wrapper: systemd-oomd is active on this box and
# kills the fit's ENTIRE pueue task-scope (bash + python together) under memory
# pressure from the user's live VS Code GPU kernel -- so the in-cgroup retry in
# u4_step3_retry.sh dies with it (observed: task 554 Killed whole, no retry line).
#
# This guard runs DETACHED in its own session/cgroup, uses ~0 RAM (so oomd never
# targets it), and launches the resumable fit ONLY into genuinely-idle GPU/RAM
# windows (see backoff below) so it never competes with the user's live work.
# jlens checkpoints every prompt (n_done monotonic: 36->45->64->69 across kills),
# so any kill only costs a model reload. Completes clean overnight once the
# user's kernel idles; does nothing (polite) while they are active.
#
# Stops itself the instant artifacts/u4_loopclose.txt exists (success). To stop
# early:  pkill -f u4_step3_guard.sh
set -u
cd /media/wassname/SGIronWolf/projects5/2026/jspace/jsteer
LABEL='U4 step3 guard-requeue'
# POLITE backoff: the user is developing jsteer live (their VS Code GPU kernel is
# the oomd competitor). Only launch the fit when the GPU/host is genuinely idle,
# so we NEVER compete with the user's interactive work -- we just fill the empty
# windows (overnight, breaks). This wastes no idle GPU time yet never fights the
# human for their own machine. Thresholds sized for the fit's ~10.5GB footprint.
GPU_FREE_MIN_MIB=13000   # need >=13GB free (i.e. others using <~11GB) to start
RAM_AVAIL_MIN_MIB=20000  # need >=20GB host available (oomd triggers on host too)
while [ ! -f artifacts/u4_loopclose.txt ]; do
  active=$(pueue status --json | jq -r --arg l "$LABEL" \
    '.tasks[] | select((.label // "") | contains($l)) | .status | if type=="object" then keys[0] else . end' \
    | grep -Ec 'Running|Queued|Stashed|Paused')
  if [ "$active" -eq 0 ]; then
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    ram_avail=$(free -m | awk '/^Mem:/{print $7}')
    if [ "$gpu_free" -ge "$GPU_FREE_MIN_MIB" ] && [ "$ram_avail" -ge "$RAM_AVAIL_MIN_MIB" ]; then
      pueue add \
        -l "why: $LABEL -- polite idle-window fit of the 4B Jacobian; resolve: resume checkpoint to n_done=400, writes u4_loopclose.txt on success" \
        -w "$PWD" -o 0 -- bash scripts/u4_step3_retry.sh
      echo "$(date '+%F %H:%M:%S') GPU idle (free=${gpu_free}MiB, ram=${ram_avail}MiB) -- launched fit"
    else
      echo "$(date '+%F %H:%M:%S') user active (gpu_free=${gpu_free}MiB, ram_avail=${ram_avail}MiB) -- backing off"
    fi
  fi
  sleep 300
done
echo "$(date '+%F %H:%M:%S') u4_loopclose.txt exists -- fit complete, guard exiting"
