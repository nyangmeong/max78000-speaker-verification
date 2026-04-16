/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the voxsv network from a template.
 * Please do not edit; instead, edit the template and regenerate.
 */

#ifndef __CNN_H__
#define __CNN_H__

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

/* Return codes */
#define CNN_FAIL 0
#define CNN_OK 1

/*
  SUMMARY OF OPS
  Hardware: 72,220,672 ops (71,811,072 macc; 296,960 comp; 112,640 add; 0 mul; 0 bitwise)
    Layer 0 (stem_Conv_10): 419,840 ops (368,640 macc; 51,200 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1 (gap_stem_pool.pool): 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2 (stage1.conv1_Conv_8): 5,939,200 ops (5,898,240 macc; 40,960 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3 (stage1.conv2_Conv_8): 5,898,240 ops (5,898,240 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4 (stage1.add): 737,280 ops (655,360 macc; 40,960 comp; 40,960 add; 0 mul; 0 bitwise)
    Layer 5 (pool1.pool): 40,960 ops (0 macc; 40,960 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6 (stage2.conv1_Conv_8): 2,969,600 ops (2,949,120 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7 (stage2.conv2_Conv_8): 5,898,240 ops (5,898,240 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8 (stage2.shortcut_Conv_8): 327,680 ops (327,680 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9 (stage2.add): 696,320 ops (655,360 macc; 20,480 comp; 20,480 add; 0 mul; 0 bitwise)
    Layer 10 (pool2.pool): 20,480 ops (0 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11 (stage3.conv1_Conv_8): 2,959,360 ops (2,949,120 macc; 10,240 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12 (stage3.conv2_Conv_8): 5,898,240 ops (5,898,240 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13 (stage3.shortcut_Conv_8): 327,680 ops (327,680 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14 (stage3.add): 675,840 ops (655,360 macc; 10,240 comp; 10,240 add; 0 mul; 0 bitwise)
    Layer 15 (stage4.conv1_Conv_8): 11,816,960 ops (11,796,480 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 16 (stage4.conv2_Conv_8): 23,592,960 ops (23,592,960 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 17 (stage4.shortcut_Conv_8): 1,310,720 ops (1,310,720 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 18 (stage4.add): 2,662,400 ops (2,621,440 macc; 20,480 comp; 20,480 add; 0 mul; 0 bitwise)
    Layer 19 (fc_Gemm_8): 28,672 ops (8,192 macc; 0 comp; 20,480 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 319,440 bytes out of 442,368 bytes total (72.2%)
  Bias memory:   784 bytes out of 2,048 bytes total (38.3%)
*/

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 32

/* Use this timer to time the inference */
#define CNN_INFERENCE_TIMER MXC_TMR0

/* Port pin actions used to signal that processing is active */

#define CNN_START LED_On(1)
#define CNN_COMPLETE LED_Off(1)
#define SYS_START LED_On(0)
#define SYS_COMPLETE LED_Off(0)

/* Run software SoftMax on unloaded data */
void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out);
/* Shift the input, then calculate SoftMax */
void softmax_shift_q17p14_q15(q31_t * vec_in, const uint16_t dim_vec, uint8_t in_shift, q15_t * p_out);

/* Stopwatch - holds the runtime when accelerator finishes */
extern volatile uint32_t cnn_time;

/* Custom memcopy routines used for weights and data */
void memcpy32(uint32_t *dst, const uint32_t *src, int n);
void memcpy32_const(uint32_t *dst, int n);

/* Enable clocks and power to accelerator, enable interrupt */
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);

/* Disable clocks and power to accelerator */
int cnn_disable(void);

/* Perform minimum accelerator initialization so it can be configured */
int cnn_init(void);

/* Configure accelerator for the given network */
int cnn_configure(void);

/* Load accelerator weights */
int cnn_load_weights(void);

/* Verify accelerator weights (debug only) */
int cnn_verify_weights(void);

/* Load accelerator bias values (if needed) */
int cnn_load_bias(void);

/* Start accelerator processing */
int cnn_start(void);

/* Force stop accelerator */
int cnn_stop(void);

/* Continue accelerator after stop */
int cnn_continue(void);

/* Unload results from accelerator */
int cnn_unload(uint32_t *out_buf);

/* Turn on the boost circuit */
int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin);

/* Turn off the boost circuit */
int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin);

#endif // __CNN_H__
