/*
 * mel_features.c
 *
 * Log-mel spectrogram computation for the voxsv speaker verification demo.
 *
 * Pipeline (matches VoxCeleb2_SV training):
 *   1. Hann-windowed STFT (win=400, hop=160, n_fft=512)
 *   2. Power spectrum
 *   3. 80-bin triangular mel filterbank (0 – 8000 Hz, HTK scale)
 *   4. Natural log, floor at 1e-6
 *   5. Per-utterance mean/variance normalisation → clip ±4σ → int8
 *
 * The Cortex-M4F FPU is used implicitly by the compiler (-mfloat-abi=softfp).
 */

#include "mel_features.h"

#include <math.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
#define MEL_PI  3.14159265358979f

/* ---- Static tables initialised in mel_init() ---- */
static float s_hann[MEL_WIN_LENGTH];          /* Hann window              */
static float s_tw_re[MEL_N_FFT / 2];          /* FFT twiddles (real part) */
static float s_tw_im[MEL_N_FFT / 2];          /* FFT twiddles (imag part) */

/* Sparse mel filterbank: triangle edges per mel bin */
static int s_mel_lo[MEL_N_MELS];
static int s_mel_pk[MEL_N_MELS];
static int s_mel_hi[MEL_N_MELS];

/* Working buffers kept off the stack */
static float s_re[MEL_N_FFT];
static float s_im[MEL_N_FFT];
static float s_power[MEL_N_FFT / 2 + 1];
static float s_logmel[MEL_N_MELS * MEL_N_FRAMES];   /* 40 KB in BSS */

/* ------------------------------------------------------------------ */
/* HTK mel ↔ Hz conversions                                           */
/* ------------------------------------------------------------------ */
static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* ------------------------------------------------------------------ */
/* In-place radix-2 DIT FFT (512-point)                               */
/* Uses precomputed twiddle table; real signal → only re/im needed.   */
/* ------------------------------------------------------------------ */
static void bit_reverse(float *re, float *im, int n)
{
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
}

static void fft(float *re, float *im, int n)
{
    bit_reverse(re, im, n);

    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        int step = n / len;   /* stride into the twiddle table */
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; k++) {
                float w_re = s_tw_re[k * step];
                float w_im = s_tw_im[k * step];
                float u_re = re[i + k];
                float u_im = im[i + k];
                float v_re = re[i + k + half] * w_re - im[i + k + half] * w_im;
                float v_im = re[i + k + half] * w_im + im[i + k + half] * w_re;
                re[i + k]        = u_re + v_re;
                im[i + k]        = u_im + v_im;
                re[i + k + half] = u_re - v_re;
                im[i + k + half] = u_im - v_im;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* mel_init                                                            */
/* ------------------------------------------------------------------ */
void mel_init(void)
{
    /* Hann window */
    for (int i = 0; i < MEL_WIN_LENGTH; i++) {
        s_hann[i] = 0.5f * (1.0f - cosf(2.0f * MEL_PI * i / (MEL_WIN_LENGTH - 1)));
    }

    /* FFT twiddle factors: W_N^k = exp(-j 2π k / N) */
    for (int k = 0; k < MEL_N_FFT / 2; k++) {
        float ang     = -2.0f * MEL_PI * k / MEL_N_FFT;
        s_tw_re[k]    =  cosf(ang);
        s_tw_im[k]    =  sinf(ang);
    }

    /* Mel filterbank: uniformly spaced in mel, converted to FFT bins */
    float mel_min = hz_to_mel(MEL_FMIN);
    float mel_max = hz_to_mel(MEL_FMAX);
    int   n_freqs = MEL_N_FFT / 2;          /* max FFT bin index (inclusive) */

    /* N_MELS + 2 mel-spaced frequency points */
    int bin_pts[MEL_N_MELS + 2];
    for (int m = 0; m < MEL_N_MELS + 2; m++) {
        float mel_f = mel_min + (float)m * (mel_max - mel_min) / (MEL_N_MELS + 1);
        float hz    = mel_to_hz(mel_f);
        int   bin   = (int)(hz / MEL_SAMPLE_RATE * MEL_N_FFT + 0.5f);
        if (bin > n_freqs) bin = n_freqs;
        bin_pts[m] = bin;
    }

    for (int m = 0; m < MEL_N_MELS; m++) {
        s_mel_lo[m] = bin_pts[m];
        s_mel_pk[m] = bin_pts[m + 1];
        s_mel_hi[m] = bin_pts[m + 2];
    }
}

/* ------------------------------------------------------------------ */
/* mel_compute                                                         */
/* ------------------------------------------------------------------ */
void mel_compute(const int16_t *audio, int8_t *out_q7)
{
    /* ---- Frame-by-frame STFT + mel filterbank ---- */
    for (int frame = 0; frame < MEL_N_FRAMES; frame++) {
        const int16_t *p = audio + frame * MEL_HOP_LENGTH;

        /* Windowed samples → FFT input (real), zero-padded to N_FFT */
        for (int k = 0; k < MEL_WIN_LENGTH; k++) {
            s_re[k] = (float)p[k] * s_hann[k] / 32768.0f;
        }
        for (int k = MEL_WIN_LENGTH; k < MEL_N_FFT; k++) {
            s_re[k] = 0.0f;
        }
        memset(s_im, 0, sizeof(s_im));

        fft(s_re, s_im, MEL_N_FFT);

        /* One-sided power spectrum: |X[k]|^2, k = 0 .. N/2 */
        for (int k = 0; k <= MEL_N_FFT / 2; k++) {
            s_power[k] = s_re[k] * s_re[k] + s_im[k] * s_im[k];
        }

        /* Triangular mel filters (librosa-style: rising=[lo,pk), falling=[pk,hi)) */
        float *row = &s_logmel[frame * MEL_N_MELS];
        for (int m = 0; m < MEL_N_MELS; m++) {
            int   lo  = s_mel_lo[m];
            int   pk  = s_mel_pk[m];
            int   hi  = s_mel_hi[m];
            float energy = 0.0f;

            /* Rising edge: weight goes from 0 at lo to 1 just before pk */
            if (pk > lo) {
                float inv = 1.0f / (float)(pk - lo);
                for (int k = lo; k < pk; k++) {
                    energy += (float)(k - lo) * inv * s_power[k];
                }
            }
            /* Falling edge: weight is 1 at pk and drops to 0 at hi */
            if (hi > pk) {
                float inv = 1.0f / (float)(hi - pk);
                for (int k = pk; k < hi; k++) {
                    energy += (float)(hi - k) * inv * s_power[k];
                }
            }

            row[m] = logf(fmaxf(energy, 1e-9f));  /* clamp matches build_dataset.py */
        }
    }

    /*
     * ---- Normalisation matching the ai8x-training VoxCeleb2FBank pipeline ----
     *
     * s_logmel layout: [MEL_N_FRAMES][MEL_N_MELS]  (frame-major)
     * out_q7   layout: [MEL_N_MELS][MEL_N_FRAMES]  (bin-major = CHW)
     *
     * Steps (must match datasets/voxceleb2.py exactly):
     *
     *   1. Per-bin temporal mean subtraction  (build_dataset.py _mean_norm)
     *        feat[t,m] -= mean_over_t(feat[:,m])
     *
     *   2. Per-segment standard-deviation normalisation
     *        std  = sqrt( mean( feat^2 ) )   [mean over all 80*128 cells]
     *        feat /= std
     *
     *   3. Clip to ±FBANK_CLIP (= 3.0)
     *
     *   4. Quantise: int8 = round( feat * (127 / FBANK_CLIP) )
     *        scale ≈ 42.33,  maps ±3.0 → ±127
     */
#define MEL_FBANK_CLIP   3.0f
#define MEL_QUANT_SCALE  (127.0f / MEL_FBANK_CLIP)   /* ≈ 42.33 */

    /* Step 1: per-bin temporal mean subtraction */
    for (int m = 0; m < MEL_N_MELS; m++) {
        float mean_m = 0.0f;
        for (int t = 0; t < MEL_N_FRAMES; t++) {
            mean_m += s_logmel[t * MEL_N_MELS + m];
        }
        mean_m /= (float)MEL_N_FRAMES;
        for (int t = 0; t < MEL_N_FRAMES; t++) {
            s_logmel[t * MEL_N_MELS + m] -= mean_m;
        }
    }

    /* Step 2: per-segment std normalisation */
    {
        float sum_sq = 0.0f;
        int   n      = MEL_N_FRAMES * MEL_N_MELS;
        for (int i = 0; i < n; i++) {
            sum_sq += s_logmel[i] * s_logmel[i];
        }
        float std = sqrtf(sum_sq / (float)n);
        if (std < 1e-9f) std = 1e-9f;   /* guard against silent frames */
        float inv_std = 1.0f / std;
        for (int i = 0; i < n; i++) {
            s_logmel[i] *= inv_std;
        }
    }

    /* Steps 3 & 4: clip ±FBANK_CLIP, quantise → int8, transpose to bin-major */
    for (int m = 0; m < MEL_N_MELS; m++) {
        for (int t = 0; t < MEL_N_FRAMES; t++) {
            float v = s_logmel[t * MEL_N_MELS + m] * MEL_QUANT_SCALE;
            if (v >  127.0f) v =  127.0f;
            if (v < -128.0f) v = -128.0f;
            out_q7[m * MEL_N_FRAMES + t] =
                (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        }
    }
}
