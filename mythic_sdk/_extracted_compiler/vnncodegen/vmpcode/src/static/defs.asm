/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2025 videantis GmbH
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
 * FILENAME:      defs.asm
 *
 * DESCRIPTION:   Global definitions for generated CNN code
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

.equ VFIR_OFFS_A = 0x376
.equ VFIR_OFFS_B = 0x377

.equ VACCU_HIGH8  = 0x2b0
.equ VACCU_MAIN8  = 0x2c0
.equ VACCU_RAW    = 0x2e0

.equ VACCU_SHIFT0 = 0x2ac
.equ VACCU_SHIFT1 = 0x2ad

// New scheme: VACCU views and shift registers
.equ VACCU_SHIFT_RD = 0x2c0
.equ VACCU_SHIFT_WR = 0x2c4
.equ VACCU_N_SHORT  = 0x2c8
.equ VACCU_W_SHORT  = 0x2cc
.equ VACCU_N_LONG   = 0x2d0
.equ VACCU_W_LONG   = 0x2d8
.equ VACCU_N_RAW    = 0x2e0
.equ VACCU_W_RAW    = 0x2f0

.equ PERMREG  = 0x388
.equ PERMREG2 = 0x38d

//FIXME use constants from lowlevel library
//      see http://vid-xc885/vid/sw/tools/vid_platform/-/issues/60
.equ VMP_WAIT_MODE_GPDATA  = {1 << 21}
.equ VMP_WAIT_MODE_DMA     = {0 << 21}
.equ VMP_WAIT_GPDATA0      = {1 <<  0}
.equ VMP_WAIT_GPDATA1      = {1 <<  1}
.equ VMP_WAIT_GPDATA2      = {1 <<  2}
.equ VMP_WAIT_GPDATA3      = {1 <<  3}
.equ VMP_WAIT_GPDATA4      = {1 <<  4}
.equ VMP_WAIT_GPDATA5      = {1 <<  5}
.equ VMP_WAIT_GPDATA6      = {1 <<  6}
.equ VMP_WAIT_GPDATA7      = {1 <<  7}
.equ VMP_WAIT_IRQOUT_SET   = {1 << 16}
.equ VMP_WAIT_IRQOUT_CLEAR = {1 << 17}
.equ VMP_WAIT_IRQIN_SET    = {1 << 18}

#define VMP_ADDR_GPDFLAGS  0x3E2


//.equ MP_VERSION= 0x3ef
