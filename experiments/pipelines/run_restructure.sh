#!/usr/bin/env bash
# Robust-restructure experiments (see kaggle_comp_report_.../ and README "Usage").
#
#   bash scripts/run_restructure.sh A      # consensus targets on the existing YOLO (5 folds, OOF)
#   bash scripts/run_restructure.sh B1     # Mask R-CNN backend (single split + sweeps)
#   bash scripts/run_restructure.sh B2     # YOLO11-seg backend (single split + sweeps)
#   bash scripts/run_restructure.sh all    # A, then B1, then B2
#
# Every command logs to work/logs/<phase>_<step>.log and the PQ lines are collected
# in work/logs/summary.txt at the end. Batch sizes are chosen automatically from the
# GPU memory free when each training starts (--batch 0), so this can be launched
# while another job still holds part of the card - it just trains smaller.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOGS=work/logs
mkdir -p "$LOGS"
SUMMARY="$LOGS/summary.txt"

run() {  # run <log-name> <command...>
    local name=$1; shift
    echo "== [$(date '+%F %T')] $*" | tee -a "$SUMMARY"
    "$@" 2>&1 | sed '/Axes3D\|warnings\.warn/d' | tee "$LOGS/$name.log"
    grep -h "PQ=" "$LOGS/$name.log" | sed "s/^/   $name: /" >> "$SUMMARY" || true
}

phase_A() {
    echo "### Phase A - consensus (majority) targets, existing axis-aligned YOLO, 5-fold OOF (reference: 0.4368)"
    run A_prepare filament prepare --k-folds 5 --augment-copy-paste 300 --annotator-merge majority
    for i in 0 1 2 3 4; do
        run A_train_yolo_fold$i filament train-yolo --fold $i --epochs 60 --batch 0
    done
    for c in 0.15 0.25 0.35; do
        run A_eval_conf$c filament evaluate --fold all --full --tta --mask-nms-thresh 0.15 --conf-thres $c
    done
}

phase_B1() {
    echo "### Phase B1 - Mask R-CNN backend (single split; downloads COCO weights ~170 MB on first run)"
    run B_prepare filament prepare --annotator-merge majority --seg
    run B1_train_maskrcnn filament train-maskrcnn --epochs 20 --batch 0
    for s in 0.5 0.6 0.7 0.8; do
        run B1_eval_score$s filament evaluate --detector maskrcnn --score-thres $s --tta --mask-nms-thresh 0.15 --full --quick 0
    done
    # extra switches at a mid threshold (re-run at the best score-thres from the sweep above if it differs)
    run B1_eval_fallback   filament evaluate --detector maskrcnn --score-thres 0.6 --tta --mask-nms-thresh 0.15 --full --quick 0 --fallback-coarse
    run B1_eval_maskconf   filament evaluate --detector maskrcnn --score-thres 0.6 --tta --mask-nms-thresh 0.15 --full --quick 0 --mask-conf-thres 0.5
    run B1_eval_norefine   filament evaluate --detector maskrcnn --score-thres 0.6 --mask-nms-thresh 0.15 --full --quick 0 --no-refine
}

phase_B2() {
    echo "### Phase B2 - YOLO11-seg backend (single split; downloads yolo11s-seg.pt ~20 MB on first run)"
    [ -f work/yolo_seg_dataset/data.yaml ] || run B_prepare filament prepare --annotator-merge majority --seg
    run B2_train_yolo_seg filament train-yolo --seg --epochs 60 --batch 0
    for c in 0.15 0.25 0.35 0.45; do
        run B2_eval_conf$c filament evaluate --detector yolo-seg --conf-thres $c --tta --mask-nms-thresh 0.15 --full --quick 0
    done
    run B2_eval_fallback filament evaluate --detector yolo-seg --conf-thres 0.25 --tta --mask-nms-thresh 0.15 --full --quick 0 --fallback-coarse
    run B2_eval_norefine filament evaluate --detector yolo-seg --conf-thres 0.25 --mask-nms-thresh 0.15 --full --quick 0 --no-refine
}

case "${1:-all}" in
    A)   phase_A ;;
    B1)  phase_B1 ;;
    B2)  phase_B2 ;;
    all) phase_A; phase_B1; phase_B2 ;;
    *)   echo "usage: $0 {A|B1|B2|all}"; exit 1 ;;
esac

echo
echo "### Summary (PQ lines, see $SUMMARY and $LOGS/*.log)"
cat "$SUMMARY"
