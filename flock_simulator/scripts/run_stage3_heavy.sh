#!/usr/bin/env bash
# Heavy Stage-3 re-runs (multi-day batch). Order: cheap first.
#
# Order: Lfine (~30 min), orderpdf_largeL (~2 h), L_highstat
# (~3-5 h), micro_hs_largeL (~3-5 h), micro_hs_L128 (~3-5 h),
# orderpdf_L128 (~24 h). The orderpdf_L128 run is the heaviest;
# we put it last so the cheaper outputs land first.
#
# Run from version4/ as:
#   nohup ./flock_simulator/scripts/run_stage3_heavy.sh \
#         > data/stage3_heavy.log 2>&1 &
set -u

cd "$(dirname "$0")/../.."
PY="../.venv/bin/python"
SCRIPTS_DIR="flock_simulator/scripts"

scripts=(
  run_double_Lfine_nocone.py
  run_double_orderpdf_largeL_nocone.py
  run_double_L_highstat_nocone.py
  run_double_micro_hs_largeL_nocone.py
  run_double_micro_hs_L128_nocone.py
  run_double_orderpdf_L128_nocone.py
)

t_start=$(date +%s)
for s in "${scripts[@]}"; do
  echo "=========================================================="
  echo "[$(date '+%F %T')] starting ${s}"
  echo "=========================================================="
  t0=$(date +%s)
  if ! "$PY" "$SCRIPTS_DIR/$s"; then
    echo "[$(date '+%F %T')] FAILED ${s}, continuing"
  fi
  t1=$(date +%s)
  echo "[$(date '+%F %T')] finished ${s} in $(( (t1-t0)/60 )) min"
done
t_end=$(date +%s)
echo "=========================================================="
echo "[$(date '+%F %T')] STAGE 3 HEAVY BATCH DONE -- total $(( (t_end-t_start)/60 )) min"
echo "=========================================================="
