#!/usr/bin/env python3
"""Patch the generated voxsv main.c with voxsv-specific helper code."""

from __future__ import annotations

import sys
from pathlib import Path

INPUT_LINE = "static const uint32_t input_0[] = SAMPLE_INPUT_0;\n"
ML_DATA_LINE = "static int32_t ml_data[CNN_NUM_OUTPUTS];\n"
UNLOAD_LINE = "  cnn_unload((uint32_t *) ml_data);\n"

PREPROCESS_BLOCK = """static const uint32_t input_0[] = SAMPLE_INPUT_0;

/*
 * Pre-processing guide for a live microphone pipeline:
 * 1. Capture mono PCM with the same sample rate used during training.
 * 2. Apply the same framing/windowing pipeline used to build VoxCeleb2_SV features.
 * 3. Convert audio to an 80-bin x 128-frame log-mel/filterbank tensor.
 * 4. Apply the same normalization and int8 quantization used during training.
 * 5. Pack the final tensor as CHW 1x80x128 and copy 2560 words into 0x50400000.
 * 6. Replace the SAMPLE_INPUT memcpy below with your computed feature buffer.
 *
 * Notes:
 * - This network expects acoustic features, not raw microphone PCM samples.
 * - Quantization/range mismatches will hurt accuracy even if the CNN runs.
 * - After inference, compare the normalized 64-D embedding with enrolled
 *   speaker embeddings using cosine similarity or dot product.
 */
"""

ML_DATA_BLOCK = """#define VOXSV_EMBEDDING_DIM 64

static int32_t ml_data[CNN_NUM_OUTPUTS];
static int8_t voxsv_embedding_q7[VOXSV_EMBEDDING_DIM];

static float voxsv_sqrt(float value)
{
  float x = value;
  int i;

  if (value <= 0.0f)
    return 0.0f;

  if (x < 1.0f)
    x = 1.0f;

  for (i = 0; i < 8; i++)
    x = 0.5f * (x + value / x);

  return x;
}

static int8_t voxsv_clamp_q7(float value)
{
  if (value > 127.0f)
    return 127;
  if (value < -128.0f)
    return -128;
  return (int8_t) (value >= 0.0f ? value + 0.5f : value - 0.5f);
}

static void voxsv_normalize_embedding_q7(const int32_t *packed_src, int8_t *dst)
{
  const int16_t *src = (const int16_t *) packed_src;
  float sum = 0.0f;
  float scale;
  int i;

  for (i = 0; i < VOXSV_EMBEDDING_DIM; i++)
    sum += (float) src[i] * (float) src[i];

  if (sum <= 0.0f) {
    memset(dst, 0, VOXSV_EMBEDDING_DIM);
    return;
  }

  scale = 128.0f / voxsv_sqrt(sum);

  for (i = 0; i < VOXSV_EMBEDDING_DIM; i++)
    dst[i] = voxsv_clamp_q7((float) src[i] * scale);
}
"""

UNLOAD_BLOCK = """  cnn_unload((uint32_t *) ml_data);
  voxsv_normalize_embedding_q7(ml_data, voxsv_embedding_q7);
  printf("Normalized embedding stored in voxsv_embedding_q7 (Q7, 64 dims).\\n");
"""


def patch_main(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False

    if "Pre-processing guide for a live microphone pipeline:" not in text:
        if INPUT_LINE not in text:
            raise RuntimeError(f"Could not find input anchor in {path}")
        text = text.replace(INPUT_LINE, PREPROCESS_BLOCK, 1)
        changed = True

    if "voxsv_normalize_embedding_q7" not in text:
        if ML_DATA_LINE not in text:
            raise RuntimeError(f"Could not find ml_data anchor in {path}")
        if UNLOAD_LINE not in text:
            raise RuntimeError(f"Could not find cnn_unload anchor in {path}")

        text = text.replace(ML_DATA_LINE, ML_DATA_BLOCK, 1)
        text = text.replace(UNLOAD_LINE, UNLOAD_BLOCK, 1)
        changed = True

    if not changed:
        return False

    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_voxsv_main.py <path-to-main.c>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    changed = patch_main(path)
    print(f"{'Patched' if changed else 'Already patched'} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
