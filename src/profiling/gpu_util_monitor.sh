OUTPUT_DIR="${1:-./profiling_results}"
DURATION="${2:-120}"

mkdir -p "$OUTPUT_DIR"
DMON_FILE="$OUTPUT_DIR/gpu_dmon_$(date +%Y%m%d_%H%M%S).log"
SNAP_FILE="$OUTPUT_DIR/gpu_snapshots_$(date +%Y%m%d_%H%M%S).log"

echo "GPU Utilization Monitor"
echo "  dmon output  : $DMON_FILE"
echo "  snap output  : $SNAP_FILE"
echo "  Duration     : ${DURATION}s"

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is an NVIDIA GPU present?"
    exit 1
fi

echo "Starting nvidia-smi dmon (SM + MEM + power, 1s interval)..."
timeout "$DURATION" nvidia-smi dmon -s pum -d 1 2>&1 \
    | tee "$DMON_FILE" &
DMON_PID=$!

echo "Starting periodic GPU snapshots (every 2s)..."
{
    ITER=0
    while [ $ITER -lt $(( DURATION / 2 )) ]; do
        echo "=== Snapshot $((ITER+1))  [$(date +%T)] ===" >> "$SNAP_FILE"
        nvidia-smi \
            --query-gpu=index,name,utilization.gpu,utilization.memory,\
memory.used,memory.free,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader 2>/dev/null >> "$SNAP_FILE"
        echo "" >> "$SNAP_FILE"
        sleep 2
        ITER=$((ITER + 1))
    done
} &
SNAP_PID=$!

echo "Monitors running. Press Ctrl-C to stop early, or wait ${DURATION}s."
wait $DMON_PID 2>/dev/null
kill $SNAP_PID 2>/dev/null

echo ""
echo "─── GPU snapshot summary ───────────────────────────────────────"
if command -v awk &>/dev/null; then
    awk -F',' '
        /^[0-9]/ {
            util    += $3 + 0
            mem_util+= $4 + 0
            mem_used+= $5 + 0
            pwr     += $8 + 0
            cnt++
        }
        END {
            if (cnt > 0) {
                printf "  Mean GPU util  : %.1f%%\n",  util/cnt
                printf "  Mean MEM util  : %.1f%%\n",  mem_util/cnt
                printf "  Mean MEM used  : %.1f MiB\n",mem_used/cnt
                printf "  Mean power     : %.1f W\n",  pwr/cnt
                printf "  Samples        : %d\n",      cnt
            }
        }
    ' "$SNAP_FILE"
fi

echo ""
echo "Done. Results:"
echo "  $DMON_FILE"
echo "  $SNAP_FILE"