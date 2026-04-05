#!/usr/bin/env bash
# GC-Bench · Run All Benchmarks
# ================================
# Runs all four benchmarks sequentially.
# Pass --data /path/to/data.h5 to specify dataset location.
# Pass --no-force to load existing checkpoints.
#
# Usage:
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --data /path/to/gc_500k.h5 --no-force

set -euo pipefail

DATA_ARG=""
FORCE_ARG=""

for arg in "$@"; do
  case $arg in
    --data=*) DATA_ARG="--data ${arg#*=}" ;;
    --data)   shift; DATA_ARG="--data $1" ;;
    --no-force) FORCE_ARG="--no-force" ;;
  esac
done

echo "============================================================"
echo "  GC-Bench: Full Benchmark Suite"
echo "  $(date)"
echo "============================================================"

echo ""
echo "▶ Benchmark 1: Forward Scalar (5 params → 4 scalars)"
python scripts/bench_forward_scalar.py $DATA_ARG $FORCE_ARG

echo ""
echo "▶ Benchmark 2: Forward Spectrum (5 params → 100-pt spectrum)"
python scripts/bench_forward_spectrum.py $DATA_ARG $FORCE_ARG

echo ""
echo "▶ Benchmark 3: Inverse Scalar (4 scalars → 5 params)"
python scripts/bench_inverse_scalar.py $DATA_ARG $FORCE_ARG

echo ""
echo "▶ Benchmark 4: Inverse Spectrum (100-pt spectrum → 5 params)"
python scripts/bench_inverse_spectrum.py $DATA_ARG $FORCE_ARG

echo ""
echo "============================================================"
echo "  All benchmarks complete."
echo "  Results saved to /kaggle/working/gc_bench/"
echo "============================================================"
