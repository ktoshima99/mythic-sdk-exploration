/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2026 videantis GmbH
 All Rights Reserved

 This document contains confidential and proprietary information of videantis
 GmbH and is protected by copyright, trade secret and other local, state,
 federal, and international laws. Its receipt or possession does not convey
 any rights to reproduce, transfer, disclose or publish its contents, or to
 manufacture, commercially or non-commercially use or sell anything it may
 describe or contain. Reproduction, disclosure or any use without specific
 written authorization of videantis GmbH or an individual license agreement
 with videantis GmbH is strictly forbidden.

 *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *
 * FILENAME:      main_defs.asm
 *
 * DESCRIPTION:   Register aliases and msg definitions for generated CNN code
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// Register aliases
.equ SFIR_OUT_DMA_DESCR     = SFIR0	// points to next out dma descr.
.equ SFIR_AUX_DMA_DESCR     = SFIR1	// points to next shortcut dma descr.
.equ SFIR_AUX_L             = SFIR2	// points to next auxiliary input (L)
.equ SFIR_AUX_R             = SFIR3	// points to next auxiliary input (R)

.equ VFIR_TMP               = VFIR0
.equ VFIR_WGT               = VFIR1
.equ VFIR_INP_L             = VFIR2
.equ VFIR_INP_R             = VFIR3
.equ VFIR_OUT_L             = VFIR4
.equ VFIR_OUT_R             = VFIR5
.equ VFIR_CDATA             = VFIR6
.equ VFIR_TMP2              = VFIR7

// Bilinear filter (uses all 8 VFIR regs)
.equ VFIR_BLF_INP_TL        = VFIR_INP_L   // must be mapped to same reg as VFIR_INP_L
.equ VFIR_BLF_INP_BL        = VFIR_INP_R   // must be mapped to same reg as VFIR_INP_R
.equ VFIR_BLF_OUT_TL        = VFIR_OUT_L   // must be mapped to same reg as VFIR_OUT_L
.equ VFIR_BLF_OUT_BL        = VFIR_OUT_R   // must be mapped to same reg as VFIR_OUT_R
.equ VFIR_BLF_INP_TR        = VFIR_TMP     // can be mapped to any free VFIR reg
.equ VFIR_BLF_INP_BR        = VFIR_WGT     // can be mapped to any free VFIR reg
.equ VFIR_BLF_OUT_TR        = VFIR_CDATA   // can be mapped to any free VFIR reg
.equ VFIR_BLF_OUT_BR        = VFIR_TMP2    // can be mapped to any free VFIR reg

// Bilinear half-pixel filter
// uses VFIR_TMP, VFIR_TMP2 and VFIR_CDATA
.equ VFIR_HP_INP_T          = VFIR_INP_L   // must be mapped to same reg as VFIR_INP_L
.equ VFIR_HP_INP_M          = VFIR_INP_R   // must be mapped to same reg as VFIR_INP_R
.equ VFIR_HP_INP_B          = VFIR_WGT     // could be mapped to any free VFIR reg, but only VFIR_WGT is free
.equ VFIR_HP_OUT_T          = VFIR_OUT_L   // must be mapped to same reg as VFIR_OUT_L
.equ VFIR_HP_OUT_B          = VFIR_OUT_R   // must be mapped to same reg as VFIR_OUT_R

.equ VR_HP_H00_H            = vr0tmp0
.equ VR_HP_H00_L            = vr0tmp1
.equ VR_HP_H10_H            = vr0tmp2
.equ VR_HP_H10_L            = vr0tmp3
.equ VR_HP_H20_H            = vr0tmp4
.equ VR_HP_H20_L            = vr0tmp5
.equ VR_HP_H01_H            = vr0tmp6
.equ VR_HP_H01_L            = vr0tmp7
.equ VR_HP_H11_H            = vr0tmp8
.equ VR_HP_H11_L            = vr0tmp9
.equ VR_HP_H21_H            = vr0tmp10
.equ VR_HP_H21_L            = vr0tmp11
.equ VR_HP_3xH10_H          = vr0tmp12
.equ VR_HP_3xH10_L          = vr0tmp13
.equ VR_HP_3xH11_H          = vr0tmp14
.equ VR_HP_3xH11_L          = vr0tmp15
.equ VR_HP_O00_H            = vr1tmp0
.equ VR_HP_O00_L            = vr1tmp1
.equ VR_HP_O10_H            = vr1tmp2
.equ VR_HP_O10_L            = vr1tmp3
.equ VR_HP_O01_H            = vr1tmp4
.equ VR_HP_O01_L            = vr1tmp5
.equ VR_HP_O11_H            = vr1tmp6
.equ VR_HP_O11_L            = vr1tmp7
.equ VR_HP_SHIFT_H          = vr1tmp8
.equ VR_HP_SHIFT_L          = vr1tmp9
.equ VR_HP_CLIP             = vr1tmp10

// SOFTMAX
.equ VR_SMAX_MAX_VAL	      = vr0tmp0
.equ VR_SMAX_SHIFT_INP      = vr0tmp1
.equ VR_SMAX_FRAC_BITS_INP  = vr0tmp2
.equ VR_SMAX_SHIFT_EXP8_0   = vr0tmp3
.equ VR_SMAX_SHIFT_EXP8_1   = vr0tmp4
.equ VR_SMAX_C0             = vr0tmp5
.equ VR_SMAX_SHIFT_C0_0     = vr0tmp6
.equ VR_SMAX_SHIFT_C0_1     = vr0tmp7
.equ VR_SMAX_MAX_INP_EXP8_0 = vr0tmp8
.equ VR_SMAX_MAX_INP_EXP8_1 = vr0tmp9
.equ VR_SMAX_IN_HI          = vr0tmp10
.equ VR_SMAX_IN_LO          = vr0tmp11
.equ VR_SMAX_MACRES_HI_HI   = vr0tmp12
.equ VR_SMAX_MACRES_HI_LO   = vr0tmp13
.equ VR_SMAX_MACRES_LO_HI   = vr0tmp14
.equ VR_SMAX_MACRES_LO_LO   = vr0tmp15

.equ VR_SMAX_FP_HI          = vr1tmp0
.equ VR_SMAX_FP_LO          = vr1tmp1
.equ VR_SMAX_INT_PART_HI    = vr1tmp2
.equ VR_SMAX_INT_PART_LO    = vr1tmp3
.equ VR_SMAX_OUT_HI         = vr1tmp4
.equ VR_SMAX_OUT_LO         = vr1tmp5
.equ VR_SMAX_XIN_HI         = vr1tmp6
.equ VR_SMAX_XIN_LO         = vr1tmp7
.equ VR_SMAX_LOG2OFE        = vr1tmp8
.equ VR_SMAX_SUM32          = vr1tmp9
.equ VR_SMAX_REG_ARF        = vr1tmp10
.equ VR_SMAX_REG_X_OUT      = vr1tmp11
.equ VR_SMAX_REG_Y_OUT      = vr1tmp12
.equ VR_SMAX_REG_Z_OUT      = vr1tmp13
.equ VR_SMAX_SHIFT_OUT      = vr1tmp14

.equ VR_SMAX_R8ACCU_HI      = vr2tmp0
.equ VR_SMAX_R8ACCU_LO      = vr2tmp1
.equ VR_SMAX_R8REG          = vr2tmp2
.equ VR_SMAX_R16ACCU_HI_HI  = vr2tmp0
.equ VR_SMAX_R16ACCU_HI_LO  = vr2tmp1
.equ VR_SMAX_R16ACCU_LO_HI  = vr2tmp2
.equ VR_SMAX_R16ACCU_LO_LO  = vr2tmp3
.equ VR_SMAX_R16REG_HI      = vr2tmp4
.equ VR_SMAX_R16REG_LO      = vr2tmp5

.equ SR_TMP0                = sparam0
.equ SR_TMP1                = sparam1
.equ SR_NUMF_CNT            = sparam2
.equ SR_NUMW_CNT            = sparam3
.equ SR_NUMH_CNT            = sparam4

.equ SR_CNT_G               = sparam2
.equ SR_CNT_NW              = sparam3
.equ SR_CNT_NH              = sparam4
.equ SR_OUT_DMA_CNT         = sparam5

.equ SR_WGT_BASE            = sr0tmp0
.equ SR_CDATA_BASE          = sr0tmp1
.equ SR_TRIGCNT_DMAOUT      = sr0tmp2
.equ SR_DMAWR_ROUTINE       = sr0tmp3
.equ SR_RETADDR_NUMH_LOOP   = sr0tmp4
.equ SR_RESH                = sr0tmp5
.equ SR_DMAOUT_EXT_ADDR     = sr0tmp6
.equ SR_DMAOUT_EXT_INC      = sr0tmp7
.equ SR_PACK_CNT0           = sr1tmp0
.equ SR_PACK_CNT1           = sr1tmp1
.equ SR_DMA_LOAD_AUX_ADDR   = sr1tmp2
.equ SR_DMA_LOAD_AUX_INC    = sr1tmp3
.equ SR_AUX_DMA_CNT         = sr1tmp4
.equ SR_AUX_BUF_CNT         = sr1tmp5
.equ SR_AUX_TOTAL_CNT       = sr1tmp6

.equ SR_MSG_CNT             = cstmp0     // GLOBAL register - will be saved when used in SDK compatible C or assembly functions

.equ SR_CNT_HH              = mstmp0
.equ SR_CNT_WW              = mstmp1

//.equ SR_WGT_BASE_2          = SR24     // for deconvolution
//.equ SR_DMAOUT_EXT_INC_2    = SR25     // for deconvolution
.equ SR_DMA_LOAD_AUX_INC_2  = mstmp2    // for deconvolution

.equ VR_TMP00               = mvtmp0
.equ VR_TMP01               = mvtmp1
.equ VR_TMP02               = mvtmp2
.equ VR_TMP03               = mvtmp3

.equ VR_PART_DIFF_MSG_0     = vr0tmp0 // no conflict with VR_RESULT_* reg usage below
.equ VR_PART_DIFF_MSG_1     = vr0tmp1 // no conflict with VR_RESULT_* reg usage below
.equ VR_PART_DIFF_MSG_2     = vr0tmp2 // no conflict with VR_RESULT_* reg usage below
.equ VR_PART_DIFF_MSG_3     = vr0tmp3 // no conflict with VR_RESULT_* reg usage below

.equ VR_RESULT_A_L          = vr0tmp0
.equ VR_RESULT_A_R          = vr0tmp1
.equ VR_RESULT_B_L          = vr0tmp2
.equ VR_RESULT_B_R          = vr0tmp3

.equ VR_RESULT_A0_L         = vr0tmp0
.equ VR_RESULT_A0_R         = vr0tmp1
.equ VR_RESULT_B0_L         = vr0tmp2
.equ VR_RESULT_B0_R         = vr0tmp3

.equ VR_RESULT_A1_L         = vr0tmp4
.equ VR_RESULT_A1_R         = vr0tmp5
.equ VR_RESULT_B1_L         = vr0tmp6
.equ VR_RESULT_B1_R         = vr0tmp7

.equ VR_AUX_A0_L            = vr0tmp8
.equ VR_AUX_A0_R            = vr0tmp9
.equ VR_AUX_B0_L            = vr0tmp10
.equ VR_AUX_B0_R            = vr0tmp11

.equ VR_DELTA_SHR_A         = vr0tmp12 // shortcut delta shr mask    |s7|s6|s5|s4|s3|s2|s1|s0|
.equ VR_DELTA_SHL_A_HI      = vr0tmp13 // shortcut delta shl mask hi |  s7 |  s6 |  s5 |  s4 |
.equ VR_DELTA_SHL_A_LO      = vr0tmp14 // shortcut delta shl mask lo |  s3 |  s2 |  s1 |  s0 |
.equ VR_DELTA_SHR_B         = vr0tmp15 // shortcut delta shr mask    |s7|s6|s5|s4|s3|s2|s1|s0|

.equ VR_DELTA_SHL_B_HI      = vr1tmp0  // shortcut delta shl mask hi |  s7 |  s6 |  s5 |  s4 |
.equ VR_DELTA_SHL_B_LO      = vr1tmp1  // shortcut delta shl mask lo |  s3 |  s2 |  s1 |  s0 |
.equ VR_MAXEXP_A            = vr1tmp2
.equ VR_MAXEXP_B            = vr1tmp3
.equ VR_SHIFTOUT_A          = vr1tmp4
.equ VR_SHIFTOUT_B          = vr1tmp5

.equ VR_RELU6_CLIPMAX_A     = vr1tmp2 // same as VR_MAXEXP_A
.equ VR_RELU6_CLIPMAX_B     = vr1tmp3 // same as VR_MAXEXP_B

.equ VR_SIGMOID_MAC0H       = mvtmp0
.equ VR_SIGMOID_MAC0L       = mvtmp1
.equ VR_SIGMOID_TMP0        = mvtmp2
.equ VR_SIGMOID_TMP1        = mvtmp3
.equ VR_SIGMOID_MINUS128    = vr1tmp6
.equ VR_SIGMOID_C0          = vr1tmp7
.equ VR_SIGMOID_C1          = vr1tmp8
.equ VR_CLIP_7F             = vr1tmp9

.equ VR_GELU_MAC0H          = mvtmp0
.equ VR_GELU_MAC0L          = mvtmp1
.equ VR_GELU_MAC1H          = mvtmp2
.equ VR_GELU_MAC1L          = mvtmp3
.equ VR_GELU_C_5B           = vr1tmp6
.equ VR_GELU_C_70           = vr1tmp7
.equ VR_GELU_C_B7           = vr1tmp8
.equ VR_GELU_C_80           = vr1tmp9

/* ========================================================================== */
// DMA channel constants

.equ dmachn_default         = 0
.equ dmachn_load_inp_first  = 1
.equ dmachn_load_inp_rest   = 2

.equ dmachn_pad_vertical    = 3
.equ dmachn_pad_left        = 4
.equ dmachn_pad_right_first = 5
.equ dmachn_pad_right_rest  = 6

.equ dmachn_store_out       = 7
.equ dmachn_store_out0      = 7
.equ dmachn_store_out1      = 8

.equ dmachn_load_aux        = 9
.equ dmachn_load_aux0       = 9
.equ dmachn_load_aux1       = 10

.equ DMA_DESCR_LOAD_AUX0    = dma_descr.start + dmachn_load_aux0 * dma_descr.CHANNELSIZE
.equ DMA_DESCR_LOAD_AUX1    = dma_descr.start + dmachn_load_aux1 * dma_descr.CHANNELSIZE
.equ WAIT_DMA_LOAD_AUX_MASK = {1 << dmachn_load_aux0} | {1 << dmachn_load_aux1}
.equ DMACTRL_LOAD_AUX0_MASK = dmachn_load_aux0 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD
.equ DMACTRL_LOAD_AUX1_MASK = dmachn_load_aux1 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD

// DMA macros for alternative input scheduling
.equ DMACTRL_LOAD_INP_REST = dmachn_load_inp_rest  |                        dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD
.equ DMACTRL_PADR_REST     = dmachn_pad_right_rest | dma_descr.DMA_MEMSET | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD
.equ DMA_DESCR_LOAD_INP_REST   = dma_descr.start + dmachn_load_inp_rest * dma_descr.CHANNELSIZE
.equ DMA_DESCR_PADR_REST       = dma_descr.start + dmachn_pad_right_rest * dma_descr.CHANNELSIZE

.equ MSG_OFS_DMA_DESCR_PADV       = MSG_OFS_DMA_DESCR_PAD0
.equ MSG_OFS_DMA_DESCR_PADL       = MSG_OFS_DMA_DESCR_PAD1
.equ MSG_OFS_DMA_DESCR_PADR_FIRST = MSG_OFS_DMA_DESCR_PAD2
.equ MSG_OFS_DMA_DESCR_PADR_REST  = MSG_OFS_DMA_DESCR_PAD3

/* ========================================================================== */
// Definitions related to msg.asm

.equ MSG_SIZE = 0x02a
.equ MSG_OFS_FLAGS__INP_LOAD_REST_TOTAL_CNT                                   = 0x000
.equ MSG_OFS_NUM_F__W__MPRES_H__NUM_H                                         = 0x001
.equ MSG_OFS_INP_BASE_L__R__SOFTPAD_FIRST__REST                               = 0x002
.equ MSG_OFS_WGTS_EXT_ADDR                                                    = 0x003
.equ MSG_OFS_CDATA_EXT_ADDR                                                   = 0x004
.equ MSG_OFS_WGTS_RSVD__LEN__CDATA_RSVD__LEN                                  = 0x005
.equ MSG_OFS_BUILD_TIMESTAMP__DEBUG_SYNC__VMP_CODE                            = 0x006
.equ MSG_OFS_DMA_DESCR_PAD0                                                   = 0x007
.equ MSG_OFS_DMA_DESCR_PAD1                                                   = 0x00a
.equ MSG_OFS_DMA_DESCR_PAD2                                                   = 0x00d
.equ MSG_OFS_DMA_DESCR_PAD3                                                   = 0x010
.equ MSG_OFS_DMA_DESCR_LOAD_INP_FIRST                                         = 0x013
.equ MSG_OFS_DMA_DESCR_LOAD_INP_REST                                          = 0x017
.equ MSG_OFS_OUT_DMATEMPLATE_STORE                                            = 0x01b
.equ MSG_OFS_OUT_STORE_EXT_BASE                                               = 0x01e
.equ MSG_OFS_OUT_STORE_EXT_INC_2__INC                                         = 0x01f
.equ MSG_OFS_OUT_STORE_INT_ADDR0__ADDR1__ADDR2__ADDR3                         = 0x020
.equ MSG_OFS_OUT_STORE_INT_ADDR4__ADDR5__BASE_L__R                            = 0x021
.equ MSG_OFS_AUX_DMATEMPLATE_LOAD                                             = 0x022
.equ MSG_OFS_AUX_LOAD_EXT_BASE                                                = 0x025
.equ MSG_OFS_AUX_LOAD_EXT_INC_2__INC                                          = 0x026
.equ MSG_OFS_AUX_LOAD_INT_ADDR0__ADDR1__ADDR2__ADDR3                          = 0x027
.equ MSG_OFS_AUX_LOAD_INT_ADDR4__ADDR5__BASE_L__R                             = 0x028
.equ MSG_OFS_OVL_EXTSRC__INTDST__LEN                                          = 0x029

.equ MSG_OFS_AVGP_THR_T__L__B__R                                              = MSG_OFS_AUX_DMATEMPLATE_LOAD // overlapping with aux


.dsection msg, dmem

.equ SIZE = MSG_SIZE // end-start does not work - why???

.equ MSGFL_WGTS_DMEM2   = 0x01
.equ MSGFL_PAD_NEG128   = 0x02

.equ SHORT_DESCR_LEN = 3

.endsection
