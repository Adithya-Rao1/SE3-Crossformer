OUTPUT_DIR="${1:-./profiling_results}"
INTERVAL="${2:-2}"     
DURATION="${3:-120}"   

mkdir -p "$OUTPUT_DIR"
OUTFILE="$OUTPUT_DIR/cpu_util_$(date +%Y%m%d_%H%M%S).log"

echo "CPU Utilization Monitor"
echo "  Output  : $OUTFILE"
echo "  Interval: ${INTERVAL}s"
echo "  Duration: ${DURATION}s"
echo "  Started : $(date)"
echo ""
echo "Interpretation guide:" | tee -a "$OUTFILE"
echo "  1 core at 100%, rest idle → Python GIL / single-threaded overhead" | tee -a "$OUTFILE"
echo "  All cores at 100%         → Data preprocessing / CPU compute bottleneck" | tee -a "$OUTFILE"
echo "  All cores idle            → Synchronisation or GPU bottleneck" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

SAMPLES=$(( DURATION / INTERVAL ))

if command -v mpstat &>/dev/null; then
    echo "Using mpstat (per-core breakdown)..." | tee -a "$OUTFILE"
    echo "─────────────────────────────────────────" | tee -a "$OUTFILE"
    mpstat -P ALL "$INTERVAL" "$SAMPLES" 2>&1 | tee -a "$OUTFILE"

elif command -v top &>/dev/null; then
    echo "mpstat not found; falling back to top..." | tee -a "$OUTFILE"
    echo "─────────────────────────────────────────" | tee -a "$OUTFILE"
    ITER=0
    while [ $ITER -lt $SAMPLES ]; do
        echo "=== Sample $((ITER+1)) / $SAMPLES  [$(date +%T)] ===" | tee -a "$OUTFILE"
        top -b -n 1 | head -20 | tee -a "$OUTFILE"
        echo "" | tee -a "$OUTFILE"
        sleep "$INTERVAL"
        ITER=$((ITER + 1))
    done

else
    echo "ERROR: Neither mpstat nor top found. Install sysstat." | tee -a "$OUTFILE"
    exit 1
fi

echo ""
echo "CPU monitoring complete. Results → $OUTFILE"

if command -v mpstat &>/dev/null && command -v awk &>/dev/null; then
    echo ""
    echo "─── Per-core average utilisation ───" | tee -a "$OUTFILE"
    grep -E "^[0-9]{2}:[0-9]{2}:[0-9]{2}.*[0-9]$" "$OUTFILE" \
        | awk '
            {
                cpu = $3
                usr = $4
                sys = $6
                tot = 100 - $13   # 100 - %idle
                sum[cpu] += tot
                cnt[cpu]++
            }
            END {
                for (c in sum) {
                    printf "  CPU %-4s : avg_util=%.1f%%\n", c, sum[c]/cnt[c]
                }
            }
        ' | sort -k2 | tee -a "$OUTFILE"
fi