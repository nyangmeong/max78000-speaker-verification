###############################################################################
# voxsv_demo project.mk
#
# Build configuration for the live speaker verification demo.
# See https://analogdevicesinc.github.io/msdk/USERGUIDE/#build-system
###############################################################################

# ---- Target board ----
BOARD ?= FTHR_RevA

# ---- Optimisation ----
MXC_OPTIMIZE_CFLAGS = -O2

# ---- Floating point ----
# softfp: FPU hardware is used for computation, but float args pass through
# integer registers – compatible with all SDK precompiled libraries.
MFLOAT_ABI = softfp

# ---- Compiler flags ----
# MEMS digital microphone (on-board, no external hardware required).
# MAX78000 acts as I2S master (MXC_I2S_INTERNAL_SCK_WS_0).

# ---- Linker flags ----
# libm provides logf, sqrtf, cosf, sinf, powf used in mel_features.c
PROJ_LDFLAGS += -lm

# ---- Project source/include layout ----
IPATH += include
VPATH += src
