#!/usr/bin/env bash
# Sequential Stage-3 batch runner for the no-cone re-runs.
#
# Order: cheap-and-quick first (snapshot, plane, orderpdf, finegrid,
# vicsek_gauss_ref) so the fast results land and any breakage is
# caught early; moderate next (profile, plane_L30, clusters_hs,
# gr_hs, decoupled); autocorr last because it is the heaviest of
# the cheap-and-moderate batch.
#
# Run from version4/ with the venv python explicitly:
#   nohup ./flock_simulator/scripts/run_stage3_batch.sh \
#         > data/stage3_batch.log 2>&1 &
set -u

cd "$(dirname "$0")/../.."   # version4/
PY="../.venv/bin/python"
SCRIPTS_DIR="flock_simulator/scripts"

scripts=(
  run_double_snapshot_nocone.py
  run_double_plane_nocone.py
  run_double_orderpdf_nocone.py
  run_double_finegrid_nocone.py
  run_vicsek_gauss_ref_nocone.py
  run_double_profile_nocone.py
  run_double_plane_L30_nocone.py
  run_double_clusters_hs_nocone.py
  run_double_gr_hs_nocone.py
  run_double_decoupled_nocone.py
  run_double_autocorr_nocone.py
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
echo "[$(date '+%F %T')] STAGE 3 BATCH DONE -- total $(( (t_end-t_start)/60 )) min"
echo "=========================================================="
