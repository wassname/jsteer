#!/usr/bin/env bash
# Claude: external requeue guard for the U4 4B Jacobian fit.
#
# WHY not just the in-task retry wrapper: systemd-oomd is active on this box and
# kills the fit's ENTIRE pueue task-scope (bash + python together) under memory
# pressure from the user's live VS Code GPU kernel -- so the in-cgroup retry in
# u4_step3_retry.sh dies with it (observed: task 554 Killed whole, no retry line).
#
# This guard runs DETACHED in its own session/cgroup, uses ~0 RAM (so oomd never
# targets it), and keeps the resumable fit queued until it completes. jlens
# checkpoints every prompt (n_done monotonic: 36->45->64->69 across kills), so
# each oomd kill only costs a model reload. It grinds slowly during the user's
# active hours and finishes clean overnight once their kernel idles.
#
# Stops itself the instant artifacts/u4_loopclose.txt exists (success). To stop
# early:  pkill -f u4_step3_guard.sh
set -u
cd /media/wassname/SGIronWolf/projects5/2026/jspace/jsteer
LABEL='U4 step3 guard-requeue'
while [ ! -f artifacts/u4_loopclose.txt ]; do
  active=$(pueue status --json | jq -r --arg l "$LABEL" \
    '.tasks[] | select((.label // "") | contains($l)) | .status | if type=="object" then keys[0] else . end' \
    | grep -Ec 'Running|Queued|Stashed|Paused')
  if [ "$active" -eq 0 ]; then
    pueue add \
      -l "why: $LABEL -- oomd-resilient external requeue of the 4B fit; resolve: resume checkpoint to n_done=400, writes u4_loopclose.txt on success" \
      -w "$PWD" -o 0 -- bash scripts/u4_step3_retry.sh
    echo "$(date '+%F %H:%M:%S') requeued fit (was not active)"
  fi
  sleep 300
done
echo "$(date '+%F %H:%M:%S') u4_loopclose.txt exists -- fit complete, guard exiting"
