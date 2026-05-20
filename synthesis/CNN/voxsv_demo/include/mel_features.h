/*
 * mel_features.h
 *
 * Log-mel spectrogram feature extraction for the voxsv speaker verification demo.
 * Parameters match the VoxCeleb2_SV training pipeline of the embedded CNN model.
 */
#pragma once
#include <stdint.h>

/* ---- STFT / mel parameters ---- */
#define MEL_SAMPLE_RATE   16000    /* Hz                               */
#define MEL_N_FFT           512    /* FFT size (window zero-padded)    */
#define MEL_WIN_LENGTH      400    /* Analysis window : 25 ms          */
#define MEL_HOP_LENGTH      160    /* Frame hop       : 10 ms          */
#define MEL_N_MELS           80    /* Mel filter banks                 */
#define MEL_N_FRAMES        128    /* Time frames                      */
#define MEL_FMIN          20.0f   /* Lowest mel frequency (Hz)  — matches build_dataset.py  */
#define MEL_FMAX        7600.0f   /* Highest mel frequency (Hz) — matches build_dataset.py  */

/*
 * Total PCM samples needed for one utterance:
 *   (N_FRAMES - 1) * HOP_LENGTH + WIN_LENGTH = 127 * 160 + 400 = 20720
 */
#define MEL_AUDIO_SAMPLES  20720

/*
 * mel_init() - Precompute Hann window, FFT twiddle factors, and mel filterbank
 *              triangle edges.  Call once at startup.
 */
void mel_init(void);

/*
 * mel_compute() - Compute the log-mel spectrogram and quantise to int8.
 *
 *   audio  : 16-bit signed PCM, exactly MEL_AUDIO_SAMPLES long, mono 16 kHz
 *   out_q7 : output [MEL_N_MELS][MEL_N_FRAMES] row-major = 10240 bytes.
 *            Represents the per-utterance MVN-normalised log-mel tensor
 *            clipped to ±4σ and scaled to the Q7 range [-128, 127].
 */
void mel_compute(const int16_t *audio, int8_t *out_q7);
