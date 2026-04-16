#!/bin/sh
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

TMP1=$(mktemp /tmp/voxsv_stage1_fakepass.XXXXXX.pth.tar)
TMP2=$(mktemp /tmp/voxsv_stage2_fakepass.XXXXXX.pth.tar)
TMP3=$(mktemp /tmp/voxsv_stage3_fakepass.XXXXXX.pth.tar)
TMP4=$(mktemp /tmp/voxsv_stage4_fakepass.XXXXXX.pth.tar)
trap 'rm -f "$TMP1" "$TMP2" "$TMP3" "$TMP4"' EXIT HUP INT TERM

"$PYTHON" izer/add_fake_passthrough.py \
    --input-checkpoint-path "$CHECKPOINT" \
    --output-checkpoint-path "$TMP1" \
    --layer-name stage1.add \
    --layer-depth 16 \
    --layer-name-after-pt stage2 \
    --low-memory-footprint

"$PYTHON" izer/add_fake_passthrough.py \
    --input-checkpoint-path "$TMP1" \
    --output-checkpoint-path "$TMP2" \
    --layer-name stage2.add \
    --layer-depth 32 \
    --layer-name-after-pt stage3 \
    --low-memory-footprint

"$PYTHON" izer/add_fake_passthrough.py \
    --input-checkpoint-path "$TMP2" \
    --output-checkpoint-path "$TMP3" \
    --layer-name stage3.add \
    --layer-depth 64 \
    --layer-name-after-pt stage4 \
    --low-memory-footprint

"$PYTHON" izer/add_fake_passthrough.py \
    --input-checkpoint-path "$TMP3" \
    --output-checkpoint-path "$TMP4" \
    --layer-name stage4.add \
    --layer-depth 128 \
    --layer-name-after-pt fc \
    --low-memory-footprint

"$PYTHON" ai8xize.py \
    --test-dir "$TARGET" \
    --prefix "$PREFIX" \
    --checkpoint-file "$TMP4" \
    --config-file "$CONFIG" \
    --board-name FTHR_RevA \
    --sample-input "$SAMPLE_INPUT" \
    --overwrite \
    $COMMON_ARGS "$@"

"$PYTHON" scripts/patch_voxsv_main.py "$TARGET/$PREFIX/main.c"
