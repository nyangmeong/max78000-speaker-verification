#!/bin/sh
# Synthesize the voxsv speaker verification model for MAX78000.
#
# Prerequisites: retrained checkpoint that includes the 1×1 conv after each
# eltwise add (ai85net-sv.py BasicBlock with split_add_relu support).
# The add_fake_passthrough.py step is NO LONGER needed — the 1×1 convs are
# already in the QAT checkpoint as stage*.add and stage4.add_relu.
set -e

PYTHON="${PYTHON:-python}"
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
PREFIX="voxsv"
CHECKPOINT="trained/ai85-voxsv-qat-q.pth.tar"
CONFIG="networks/sv.yaml"
SAMPLE_INPUT="tests/sample_voxceleb2_sv.npy"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose --no-version-check"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

if [ ! -f "$SAMPLE_INPUT" ]; then
    echo "Sample input not found: $SAMPLE_INPUT" >&2
    exit 1
fi

"$PYTHON" ai8xize.py \
    --test-dir "$TARGET" \
    --prefix "$PREFIX" \
    --checkpoint-file "$CHECKPOINT" \
    --board-name FTHR_RevA \
    --config-file "$CONFIG" \
    --sample-input "$SAMPLE_INPUT" \
    --overwrite \
    $COMMON_ARGS "$@"

"$PYTHON" scripts/patch_voxsv_main.py "$TARGET/$PREFIX/main.c"
