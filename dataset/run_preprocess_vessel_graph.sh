#!/bin/bash
# Full run of dataset/preprocess_vessel_graph.py over every fitted case, N workers in parallel.
# Output: runtime_dataset/assembled/<dataset>/<case>/vessel_graph/   (skips cases already done)
# Work:   runtime_vessel_graph_work/<dataset>/<case>/, logs in runtime_vessel_graph_work/logs/
# A worker waits before starting a case while MemAvailable < MIN_FREE_GB (peak ~10 GB per case).
#   setsid nohup bash dataset/run_preprocess_vessel_graph.sh 8 > runtime_vessel_graph_work/launcher.log 2>&1 &
WORKERS=${1:-8}
MIN_FREE_GB=${MIN_FREE_GB:-15}
REPO="/media/yaplab2/HDD Storage/wenhao/AneuG"
PY=/home/yaplab2/miniconda3/envs/new/bin/python
WORK="$REPO/runtime_vessel_graph_work"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$REPO" || exit 1
# one queue, ImperialNHS and AneuX alternating (AneuX cases are the heavy ones)
$PY - > "$WORK/queue.txt" <<'PYEOF'
import os
A = "runtime_dataset/assembled"
imp = sorted(c for c in os.listdir(f"{A}/ImperialNHS") if os.path.isdir(f"{A}/ImperialNHS/{c}"))
anx = sorted(c for c in os.listdir(f"{A}/AneuX") if os.path.isdir(f"{A}/AneuX/{c}"))
out = []
for i in range(max(len(imp), len(anx))):
    if i < len(anx): out.append(f"AneuX {anx[i]}")
    if i < len(imp): out.append(f"ImperialNHS {imp[i]}")
print("\n".join(out))
PYEOF
echo "$(date '+%F %T') start: $(wc -l < "$WORK/queue.txt") cases, $WORKERS workers, min free ${MIN_FREE_GB} GB"
export PY WORK MIN_FREE_GB
xargs -P "$WORKERS" -L 1 bash -c '
  ds=$0; case=$1
  while [ "$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)" -lt "$MIN_FREE_GB" ]; do sleep 30; done
  "$PY" dataset/preprocess_vessel_graph.py --dataset "$ds" --cases "$case" \
      --work_root "$WORK" > "$WORK/logs/${ds}__${case}.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "$(date "+%F %T") $ds $case exited $rc"
' < "$WORK/queue.txt"
echo "$(date '+%F %T') finished"
