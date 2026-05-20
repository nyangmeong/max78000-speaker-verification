/*
 * voxsv_demo/main.c
 *
 * Continuous VAD-driven speaker embedding demo — MAX78000 FTHR_RevA.
 *
 * Demo flow
 * ---------
 *   Just speak.  The demo auto-detects speech onset, records ~1.3 s,
 *   extracts a 64-D speaker embedding, and compares it with the
 *   previous one.  No button press required.
 *
 *   First utterance  → stored as reference (no comparison)
 *   Each subsequent  → cosine similarity vs previous + SAME/DIFF result
 *
 * Hardware
 * --------
 *   MEMS digital microphone  on-board, I2S internal clock, no extra HW
 *   LED0 (red)               recording active
 *   LED1 (green)             listening / SAME-SPEAKER flash
 *   SW1                      (optional) force re-arm / reset reference
 *
 * CNN backend
 * -----------
 *   ai85-voxsv-qat-q  ThinResNet  →  64-D L2-normalised d-vector (Q7)
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "nvic_table.h"
#include "icc.h"
#include "board.h"
#include "led.h"
#include "pb.h"
#include "i2s.h"
#include "i2s_regs.h"

#include "cnn.h"
#include "mel_features.h"
#include "speaker_db.h"

/* ------------------------------------------------------------------ */
/* Hardware / audio parameters                                          */
/* ------------------------------------------------------------------ */
#define SAMPLE_RATE          16000
#define EXT_I2S_FREQ         12288000
#define I2S_RX_BUFFER_SIZE   64

/* Chunk size for energy computation — one I2S ISR batch worth         */
#define CHUNK                128

/* Voice-activity thresholds (mean-absolute int16 amplitude post-HPF)  */
/* kws20_demo uses 350 / 100 on FTHR_RevA — same hardware, same mic   */
#define VAD_THRESHOLD_HIGH   350
#define VAD_THRESHOLD_LOW    100

/* Warmup: discard first N samples after mic power-on (charging cap)   */
#define MIC_WARMUP_SAMPLES   10000

/* Cooldown: drain mic for this many samples after each inference       */
/* to prevent the same utterance triggering twice (~1.5 s)             */
#define COOLDOWN_SAMPLES     (SAMPLE_RATE + SAMPLE_RATE / 2)   /* 24000 */

/* ------------------------------------------------------------------ */
/* Embedding / verification                                             */
/* ------------------------------------------------------------------ */
#define EMBEDDING_DIM     64
#define VERIFY_THRESHOLD  0.80f

/* ------------------------------------------------------------------ */
/* Speaker database (populated at build time by db_gen)                 */
/* ------------------------------------------------------------------ */
#if DB_NUM_ENTRIES > 0
static const char   g_db_names[DB_NUM_ENTRIES][7] = DB_SPEAKER_NAMES;
static const int8_t g_db_embs [DB_NUM_ENTRIES][DB_EMBEDDING_DIM] = DB_EMBEDDINGS;
#endif

/* ================================================================== */
/* Globals                                                              */
/* ================================================================== */
volatile uint32_t cnn_time;

static int8_t  g_prev[EMBEDDING_DIM];
static int8_t  g_current[EMBEDDING_DIM];
static int     g_prev_valid = 0;
static int32_t g_ml_data[CNN_NUM_OUTPUTS];

static int16_t  g_audio[MEL_AUDIO_SAMPLES];
static uint32_t g_features_u32[MEL_N_MELS * MEL_N_FRAMES / 4];

static int32_t g_i2s_rx[I2S_RX_BUFFER_SIZE];
static volatile int g_i2s_flag = 0;

/* ================================================================== */
/* I2S interrupt handler                                                */
/* ================================================================== */
static void i2s_isr(void)
{
    g_i2s_flag = 1;
    MXC_I2S_ClearFlags(MXC_F_I2S_INTFL_RX_THD_CH0);
}

/* ================================================================== */
/* High-pass filter  y[n] = x[n] - x[n-1] + 0.995*y[n-1]             */
/* ================================================================== */
#define HPF_COEFF 32604
static int16_t hpf_x1;
static int32_t hpf_y1;

static void hpf_reset(void)  { hpf_x1 = 0; hpf_y1 = 0; }

static int16_t hpf(int16_t x)
{
    int32_t y = (int32_t)x - hpf_x1 + ((HPF_COEFF * hpf_y1 + (1 << 14)) >> 15);
    if (y >  32767) y =  32767;
    if (y < -32768) y = -32768;
    hpf_x1 = x;
    hpf_y1 = y;
    return (int16_t)y;
}

/* ================================================================== */
/* I2S init                                                             */
/* ================================================================== */
static void i2s_init(void)
{
    mxc_i2s_req_t req;
    memset(g_i2s_rx, 0, sizeof(g_i2s_rx));

    req.wordSize    = MXC_I2S_WSIZE_WORD;
    req.sampleSize  = MXC_I2S_SAMPLESIZE_THIRTYTWO;
    req.bitsWord    = 32;
    req.adjust      = MXC_I2S_ADJUST_LEFT;
    req.justify     = MXC_I2S_MSB_JUSTIFY;
    req.wsPolarity  = MXC_I2S_POL_NORMAL;
    req.channelMode = MXC_I2S_INTERNAL_SCK_WS_0;  /* MAX78000 = I2S master, MEMS mic */
    req.stereoMode  = MXC_I2S_MONO_LEFT_CH;
    req.bitOrder    = MXC_I2S_MSB_FIRST;
    req.clkdiv      = MXC_I2S_CalculateClockDiv(SAMPLE_RATE, req.wordSize, EXT_I2S_FREQ);
    req.rawData     = NULL;
    req.txData      = NULL;
    req.rxData      = g_i2s_rx;
    req.length      = I2S_RX_BUFFER_SIZE;

    if (MXC_I2S_Init(&req) != E_NO_ERROR) {
        printf("[ERR] I2S init failed – halting\n");
        while (1) {}
    }

    MXC_I2S_SetRXThreshold(4);
    MXC_NVIC_SetVector(I2S_IRQn, i2s_isr);
    NVIC_EnableIRQ(I2S_IRQn);
    MXC_I2S_EnableInt(MXC_F_I2S_INTEN_RX_THD_CH0);
    MXC_I2S_RXEnable();
    __enable_irq();
}

/* ================================================================== */
/* Mic drain helpers                                                    */
/*                                                                      */
/* drain_fifo():  reads whatever is in the FIFO right now.             */
/*   capture_dst  – if non-NULL and *idx < max_idx, stores samples.   */
/*   Returns mean absolute amplitude of samples read (0 if flag off). */
/* ================================================================== */
static uint16_t drain_fifo(int16_t *capture_dst, int *idx, int max_idx)
{
    if (!g_i2s_flag) return 0;
    g_i2s_flag = 0;

    uint32_t rx_lvl = MXC_I2S->dmach0 >> MXC_F_I2S_DMACH0_RX_LVL_POS;
    uint32_t sum    = 0;
    uint32_t count  = 0;

    while (rx_lvl--) {
        int32_t raw = (int32_t)MXC_I2S->fifoch0;
        int16_t pcm = hpf((int16_t)(raw >> 14));
        int16_t ab  = pcm < 0 ? -pcm : pcm;
        sum += (uint32_t)ab;
        count++;

        /* Always increment idx (drain/cooldown/warmup counting).
         * Store only when capture_dst is provided.               */
        if (*idx < max_idx) {
            if (capture_dst)
                capture_dst[*idx] = pcm;
            (*idx)++;
        }
    }

    return (count > 0) ? (uint16_t)(sum / count) : 0;
}

/* ================================================================== */
/* CNN inference                                                         */
/* ================================================================== */
static void run_cnn(void)
{
    memcpy32((uint32_t *)0x50400000, g_features_u32, MEL_N_MELS * MEL_N_FRAMES / 4);
    cnn_time = 0;
    cnn_configure();
    cnn_start();
    while (cnn_time == 0) { MXC_LP_EnterSleepMode(); }
    cnn_unload((uint32_t *)g_ml_data);
}

/* ================================================================== */
/* L2-normalise packed int16 CNN output → Q7 int8 embedding             */
/* ================================================================== */
static void normalize_embedding(const int32_t *packed, int8_t *dst)
{
    const int16_t *src = (const int16_t *)packed;
    float sum = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; i++)
        sum += (float)src[i] * (float)src[i];
    if (sum <= 0.0f) { memset(dst, 0, EMBEDDING_DIM); return; }
    float scale = 128.0f / sqrtf(sum);
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        float v = (float)src[i] * scale;
        if (v >  127.0f) v =  127.0f;
        if (v < -128.0f) v = -128.0f;
        dst[i] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    }
}

/* ================================================================== */
/* Cosine similarity between two Q7 unit-norm embeddings               */
/* ================================================================== */
static float cosine_similarity(const int8_t *a, const int8_t *b)
{
    int32_t dot = 0;
    for (int i = 0; i < EMBEDDING_DIM; i++)
        dot += (int32_t)a[i] * (int32_t)b[i];
    return (float)dot / (128.0f * 128.0f);
}

/* ================================================================== */
/* DB identification (1:N)                                              */
/* ================================================================== */
#if DB_NUM_ENTRIES > 0
static int db_identify(const int8_t *query, float *best_sim)
{
    int   best_idx = -1;
    float best     = -1.0f;
    for (int i = 0; i < DB_NUM_ENTRIES; i++) {
        float s = cosine_similarity(query, g_db_embs[i]);
        if (s > best) { best = s; best_idx = i; }
    }
    *best_sim = best;
    return (best >= VERIFY_THRESHOLD) ? best_idx : -1;
}
#endif


/* ================================================================== */
/* DB build mode                                                        */
/*                                                                      */
/* Activated by holding SW1 within 3 s after "[INIT] Done".            */
/* VAD → record → mel → CNN → normalize → send embedding over UART.   */
/* Use db_gen/build_db.py on the PC to collect and write speaker_db.h  */
/*                                                                      */
/* Protocol (one utterance):                                            */
/*   "READY\n"                          board ready, green LED on       */
/*   "EMBED:<64 comma-sep int8>\n"      embedding sent after inference  */
/* ================================================================== */
static void db_build_mode(void)
{
    mxc_uart_regs_t *uart = MXC_UART_GET_UART(CONSOLE_UART);

    printf("===========================================\n");
    printf("  DB BUILD MODE\n");
    printf("  Run: python db_gen/manage_db.py\n");
    printf("  Waiting for PC command...\n");
    printf("===========================================\n\n");

    for (;;) {
        /* Wait for 'G' (GO) from PC before each utterance.
         * Drain I2S FIFO meanwhile to prevent overflow.       */
        {
            int dummy = 0;
            int c = -1;
            while (c != 'G') {
                drain_fifo(NULL, &dummy, 0);
                c = MXC_UART_ReadCharacter(uart);
            }
        }

        printf("READY\n");
        LED_On(LED_GREEN);

        /* Wait for voice onset */
        for (;;) {
            int dummy = 0;
            uint16_t avg = drain_fifo(NULL, &dummy, 0);
            if (avg >= VAD_THRESHOLD_HIGH) break;
        }
        LED_Off(LED_GREEN);

        /* Capture */
        LED_On(LED_RED);
        int idx = 0;
        while (idx < MEL_AUDIO_SAMPLES)
            drain_fifo(g_audio, &idx, MEL_AUDIO_SAMPLES);
        LED_Off(LED_RED);

        /* Level info for PC script quality feedback */
        {
            int32_t sq = 0;
            int16_t pk = 0;
            for (int i = 0; i < MEL_AUDIO_SAMPLES; i++) {
                int16_t s = g_audio[i] < 0 ? -g_audio[i] : g_audio[i];
                if (s > pk) pk = s;
                sq += (int32_t)g_audio[i] * g_audio[i] / MEL_AUDIO_SAMPLES;
            }
            printf("LEVEL peak=%-5d rms=%-5d\n", (int)pk, (int)sqrtf((float)sq));
        }

        /* Mel + CNN + embedding */
        mel_compute(g_audio, (int8_t *)g_features_u32);
        run_cnn();
        normalize_embedding(g_ml_data, g_current);

        /* Send embedding */
        printf("EMBED:");
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            printf("%d", (int)g_current[i]);
            if (i < EMBEDDING_DIM - 1) printf(",");
        }
        printf("\n");

        /* Cooldown */
        int cd = 0;
        while (cd < COOLDOWN_SAMPLES)
            drain_fifo(NULL, &cd, COOLDOWN_SAMPLES);
    }
}

/* ================================================================== */
/* LED flash                                                            */
/* ================================================================== */
static void led_flash(int led, int times)
{
    for (int i = 0; i < times; i++) {
        LED_On(led);  MXC_Delay(MXC_DELAY_MSEC(200));
        LED_Off(led); MXC_Delay(MXC_DELAY_MSEC(150));
    }
}

/* ================================================================== */
/* main                                                                 */
/* ================================================================== */
int main(void)
{
    /* ---- System clock ---- */
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    MXC_GCR->pm &= ~MXC_F_GCR_PM_IPO_PD;
    SystemCoreClockUpdate();

    /* Wait for PMIC 1.8V rail */
    MXC_Delay(MXC_DELAY_MSEC(200));

    /* ---- Banner ---- */
    printf("\n");
    printf("===========================================\n");
    printf("  VoxSV Speaker Verification Demo\n");
    printf("  MAX78000 FTHR_RevA\n");
    printf("===========================================\n");
    printf("  Threshold : %.2f  |  Embedding : %d-D\n",
           (double)VERIFY_THRESHOLD, EMBEDDING_DIM);
    printf("  Audio     : %d ms  |  VAD high  : %d\n",
           MEL_AUDIO_SAMPLES * 1000 / SAMPLE_RATE, VAD_THRESHOLD_HIGH);
    printf("===========================================\n\n");

    /* ---- MEMS mic power — MUST be before I2S init ---- */
    printf("[INIT] MEMS mic power...\n");
    Microphone_Power(POWER_ON);

    printf("[INIT] I2S...\n");
    i2s_init();

    printf("[INIT] CNN accelerator...\n");
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();

    printf("[INIT] Mel filterbank...\n");
    mel_init();

    /* ---- One-time mic warmup: drain first MIC_WARMUP_SAMPLES ---- */
    printf("[INIT] Mic warmup (%d samples)...\n", MIC_WARMUP_SAMPLES);
    hpf_reset();
    {
        int dummy = 0;
        while (dummy < MIC_WARMUP_SAMPLES) {
            int before = dummy;
            drain_fifo(NULL, &dummy, MIC_WARMUP_SAMPLES);
            (void)before;
        }
    }

    printf("[INIT] Done.\n\n");
    led_flash(LED_GREEN, 2);

    /* ---- DB build mode: press SW1 within 3 s after init ----
     * LEDs alternate red/green while waiting.
     * Any SW1 press during the window → enter DB build mode.        */
    printf("  Press SW1 within 3 s to enter DB BUILD mode...\n\n");
    {
        int enter_db_build = 0;
        for (int t = 0; t < 30 && !enter_db_build; t++) {
            LED_On(t & 1 ? LED_GREEN : LED_RED);
            MXC_Delay(MXC_DELAY_MSEC(100));
            LED_Off(t & 1 ? LED_GREEN : LED_RED);
            if (PB_Get(0)) enter_db_build = 1;
        }
        if (enter_db_build) {
            db_build_mode();   /* never returns */
        }
    }

/* ================================================================== */
#if DB_NUM_ENTRIES > 0
    /* ---- DB IDENTIFICATION MODE (1:N) -------------------------------- */
    printf("===========================================\n");
    printf("  DB mode: %d speaker(s) loaded\n", DB_NUM_ENTRIES);
    for (int i = 0; i < DB_NUM_ENTRIES; i++)
        printf("    [%d] %s\n", i, g_db_names[i]);
    printf("===========================================\n\n");

    for (;;) {
        printf("  [VAD] Listening...  (speak!)\n");
        LED_On(LED_GREEN);

        /* Wait for voice onset */
        for (;;) {
            int dummy = 0;
            uint16_t avg = drain_fifo(NULL, &dummy, 0);
            if (avg >= VAD_THRESHOLD_HIGH) break;
        }
        LED_Off(LED_GREEN);

        /* Capture */
        printf("  [REC] Capturing %.1f s...\n", (double)MEL_AUDIO_SAMPLES / SAMPLE_RATE);
        LED_On(LED_RED);
        int idx = 0;
        while (idx < MEL_AUDIO_SAMPLES)
            drain_fifo(g_audio, &idx, MEL_AUDIO_SAMPLES);
        LED_Off(LED_RED);

        /* Level diagnostic */
        int32_t sq = 0;
        int16_t pk = 0;
        for (int i = 0; i < MEL_AUDIO_SAMPLES; i++) {
            int16_t s = g_audio[i] < 0 ? -g_audio[i] : g_audio[i];
            if (s > pk) pk = s;
            sq += (int32_t)g_audio[i] * g_audio[i] / MEL_AUDIO_SAMPLES;
        }
        printf("  [REC] done.  peak=%-5d  rms=%-5d\n", (int)pk, (int)sqrtf((float)sq));

        /* Mel + CNN */
        printf("  [MEL] Computing log-mel...\n");
        mel_compute(g_audio, (int8_t *)g_features_u32);
        printf("  [CNN] Running inference...\n");
        run_cnn();
        printf("  [CNN] %u us\n", cnn_time);

        normalize_embedding(g_ml_data, g_current);

        float best_sim;
        int match = db_identify(g_current, &best_sim);

        printf("\n  +------------------------------------+\n");
        printf("  | Similarity : %+.4f               |\n", (double)best_sim);
        if (match >= 0) {
            printf("  | Result     : ACCEPTED  :)         |\n");
            printf("  | Speaker    : %-6s                |\n", g_db_names[match]);
            printf("  +------------------------------------+\n\n");
            led_flash(LED_GREEN, 4);
        } else {
            printf("  | Result     : REJECTED  :(         |\n");
            printf("  +------------------------------------+\n\n");
            led_flash(LED_RED, 4);
        }

        /* Cooldown */
        int cd = 0;
        while (cd < COOLDOWN_SAMPLES)
            drain_fifo(NULL, &cd, COOLDOWN_SAMPLES);
    }

/* ================================================================== */
#else
    /* ---- CONTINUOUS COMPARE MODE ------------------------------------
     *
     * Utterance #1  → stored as reference, no comparison
     * Utterance #N  → cosine similarity vs utterance #N-1
     *
     * Use this to calibrate VERIFY_THRESHOLD:
     *   Same speaker across utterances  → expect > 0.75
     *   Different speakers              → expect < 0.50
     * ---------------------------------------------------------------- */
    int iter = 0;

    for (;;) {
        /* Listening prompt */
        printf("-------------------------------------------\n");
        if (!g_prev_valid)
            printf("  [VAD] Listening...  speak for recording #1\n");
        else
            printf("  [VAD] Listening...  speak to compare with #%d\n", iter);
        printf("-------------------------------------------\n");
        LED_On(LED_GREEN);

        /* ---- VAD: wait for speech onset ---- */
        for (;;) {
            int dummy = 0;
            uint16_t avg = drain_fifo(NULL, &dummy, 0);
            if (avg >= VAD_THRESHOLD_HIGH) break;
        }
        LED_Off(LED_GREEN);

        /* ---- Capture exactly MEL_AUDIO_SAMPLES ---- */
        printf("  [REC] Capturing %.1f s...\n", (double)MEL_AUDIO_SAMPLES / SAMPLE_RATE);
        LED_On(LED_RED);
        int idx = 0;
        while (idx < MEL_AUDIO_SAMPLES)
            drain_fifo(g_audio, &idx, MEL_AUDIO_SAMPLES);
        LED_Off(LED_RED);

        /* ---- Level diagnostic ---- */
        int32_t sq = 0;
        int16_t pk = 0;
        for (int i = 0; i < MEL_AUDIO_SAMPLES; i++) {
            int16_t s = g_audio[i] < 0 ? -g_audio[i] : g_audio[i];
            if (s > pk) pk = s;
            sq += (int32_t)g_audio[i] * g_audio[i] / MEL_AUDIO_SAMPLES;
        }
        printf("  [REC] done.  peak=%-5d  rms=%-5d  (silence<100, speech>500)\n",
               (int)pk, (int)sqrtf((float)sq));

        /* ---- Mel ---- */
        printf("  [MEL] Computing log-mel...\n");
        mel_compute(g_audio, (int8_t *)g_features_u32);

        /* ---- CNN ---- */
        printf("  [CNN] Running inference...\n");
        run_cnn();
        printf("  [CNN] %u us\n", cnn_time);

        /* ---- Embedding ---- */
        normalize_embedding(g_ml_data, g_current);
        iter++;

        /* ---- Compare ---- */
        if (g_prev_valid) {
            float sim = cosine_similarity(g_prev, g_current);

            printf("\n  +------------------------------------+\n");
            printf("  | #%-3d vs #%-3d                      |\n", iter - 1, iter);
            printf("  | Similarity : %+.4f               |\n", (double)sim);
            if (sim >= VERIFY_THRESHOLD) {
                printf("  | Result     : SAME SPEAKER  :)     |\n");
                printf("  +------------------------------------+\n\n");
                led_flash(LED_GREEN, 3);
            } else {
                printf("  | Result     : DIFF SPEAKER  :(     |\n");
                printf("  +------------------------------------+\n\n");
                led_flash(LED_RED, 3);
            }
        } else {
            printf("\n  [OK] Recording #1 stored.  Now speak to compare.\n\n");
            led_flash(LED_GREEN, 1);
        }

        /* Current → previous */
        memcpy(g_prev, g_current, EMBEDDING_DIM);

        g_prev_valid = 1;

        /* ---- Cooldown: drain mic for ~1.5 s to prevent re-trigger ---- */
        int cd = 0;
        while (cd < COOLDOWN_SAMPLES)
            drain_fifo(NULL, &cd, COOLDOWN_SAMPLES);
    }
#endif
}
