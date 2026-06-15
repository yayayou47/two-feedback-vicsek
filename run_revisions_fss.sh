#!/usr/bin/env bash
# Orchestrated launcher for the heavy revision FSS runs A1 + D1.
#   A1  : L = 180, 256 at rho0 = 2.22
#   D1a : rho0 = 1.0, L = 22..128
#   D1b : rho0 = 3.0, L = 22..128
# Sequential (avoids oversubscribing 8 cores), each cell single-numba-thread
# so the multiprocessing pool of 7 workers owns the parallelism. The driver
# is resumable cell-by-cell, so an interruption just restarts where it left.
set -u
cd "$(dirname "$0")"
PY="../.venv/bin/python"
DRV="flock_simulator/scripts/run_fss_parallel_nocone.py"
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
W=7
LOG="data/revisions_fss.log"

acline() { cat /sys/class/power_supply/AC*/online 2>/dev/null | head -1; }

echo "=== launcher start $(date -Iseconds) ===" | tee -a "$LOG"
# Power guard disabled: user opted to run on battery. The driver checkpoints
# every 40 cells and resumes from disk, so a battery cutoff loses no finished
# work. AC line is logged for the record only.
echo "AC=$(acline) (battery run accepted), starting $(date -Iseconds)" | tee -a "$LOG"

run() {
  local tag="$1"; shift
  echo "--- $tag start $(date -Iseconds) ---" | tee -a "$LOG"
  "$PY" "$DRV" "$@" --workers "$W" 2>&1 | tee -a "$LOG"
  echo "--- $tag done $(date -Iseconds) ---" | tee -a "$LOG"
}

run A1  --rho 2.22 --sizes 180,256 \
        --out double_fss_homog10_largeL_nocone.npz
run D1a --rho 1.0  --sizes 22,30,45,64,90,128 \
        --out double_fss_density_rho1p0_nocone.npz
run D1b --rho 3.0  --sizes 22,30,45,64,90,128 \
        --out double_fss_density_rho3p0_nocone.npz

echo "=== launcher all done $(date -Iseconds) ===" | tee -a "$LOG"
