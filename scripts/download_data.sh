#!/usr/bin/env bash
# Downloads the Filament Segmentation Challenge 2026 dataset via the Kaggle API
# into FILAMENT_DATA_ROOT's parent (default: /data/marchezi/filament_segmentation_2026).
#
# Requires ~/.kaggle/kaggle.json (see README.md > Getting the data).
set -euo pipefail

DEST_PARENT="${1:-/data/marchezi/filament_segmentation_2026}"
mkdir -p "$DEST_PARENT"
cd "$DEST_PARENT"

kaggle competitions download -c filament-segmentation-2026
unzip -q filament-segmentation-2026.zip
rm filament-segmentation-2026.zip

echo "Dataset ready at $DEST_PARENT/MAGFiLO_1.0_Kaggle_2026"
