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
 * FILENAME:    vid_vmp_ioregs.asm
 *
 * DESCRIPTION: Address definitions of v-MP internal IO registers
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis LowLevel Library vid_vmp_ioregs.asm include file
 *
 * @details
 * Address definitions of v-MP internal IO registers (assembler equivalent
 * of vid_vmp_ioregs.h).
 *
 * @file vid_vmp_ioregs.asm
 */

/// @cond DOXYGEN_IGNORE_ASM

.equ VMP_ADDR_BOOT_ADDR_L = 0x3BE
.equ VMP_ADDR_BOOT_ADDR_H = 0x3BF

.equ VMP_ADDR_TIMER1    = 0x3D0
.equ VMP_ADDR_TCTRL1    = 0x3D1
.equ VMP_ADDR_TIMER2    = 0x3D2
.equ VMP_ADDR_TCTRL2    = 0x3D3

.equ VMP_ADDR_IRQIC     = 0x3D6
.equ VMP_ADDR_IRQIN     = 0x3D8
.equ VMP_ADDR_IRQOUT    = 0x3D9

.equ VMP_ADDR_GPDATA0   = 0x3DA
.equ VMP_ADDR_GPDATA1   = 0x3DB
.equ VMP_ADDR_GPDATA2   = 0x3DC
.equ VMP_ADDR_GPDATA3   = 0x3DD
.equ VMP_ADDR_GPDATA4   = 0x3DE
.equ VMP_ADDR_GPDATA5   = 0x3DF
.equ VMP_ADDR_GPDATA6   = 0x3E0
.equ VMP_ADDR_GPDATA7   = 0x3E1
.equ VMP_ADDR_GPD_FLAGS = 0x3E2
.equ VMP_ADDR_GPDFLAGS  = 0x3E2


// Byte-address views of the IO registers (one word = 8 bytes).
.equ VMP_BADDR_BOOT_ADDR_L = VMP_ADDR_BOOT_ADDR_L << 3
.equ VMP_BADDR_BOOT_ADDR_H = VMP_ADDR_BOOT_ADDR_H << 3

.equ VMP_BADDR_TIMER1    = VMP_ADDR_TIMER1 << 3
.equ VMP_BADDR_TCTRL1    = VMP_ADDR_TCTRL1 << 3
.equ VMP_BADDR_TIMER2    = VMP_ADDR_TIMER2 << 3
.equ VMP_BADDR_TCTRL2    = VMP_ADDR_TCTRL2 << 3

.equ VMP_BADDR_IRQIC     = VMP_ADDR_IRQIC  << 3
.equ VMP_BADDR_IRQIN     = VMP_ADDR_IRQIN  << 3
.equ VMP_BADDR_IRQOUT    = VMP_ADDR_IRQOUT << 3

.equ VMP_BADDR_GPDATA0   = VMP_ADDR_GPDATA0   << 3
.equ VMP_BADDR_GPDATA1   = VMP_ADDR_GPDATA1   << 3
.equ VMP_BADDR_GPDATA2   = VMP_ADDR_GPDATA2   << 3
.equ VMP_BADDR_GPDATA3   = VMP_ADDR_GPDATA3   << 3
.equ VMP_BADDR_GPDATA4   = VMP_ADDR_GPDATA4   << 3
.equ VMP_BADDR_GPDATA5   = VMP_ADDR_GPDATA5   << 3
.equ VMP_BADDR_GPDATA6   = VMP_ADDR_GPDATA6   << 3
.equ VMP_BADDR_GPDATA7   = VMP_ADDR_GPDATA7   << 3
.equ VMP_BADDR_GPD_FLAGS = VMP_ADDR_GPD_FLAGS << 3
.equ VMP_BADDR_GPDFLAGS  = VMP_ADDR_GPDFLAGS  << 3


.equ VMP_WAIT_MODE_GPDATA  = 1 << 21
.equ VMP_WAIT_MODE_DMA     = 0 << 21
.equ VMP_WAIT_GPDATA0      = 1 <<  0
.equ VMP_WAIT_GPDATA1      = 1 <<  1
.equ VMP_WAIT_GPDATA2      = 1 <<  2
.equ VMP_WAIT_GPDATA3      = 1 <<  3
.equ VMP_WAIT_GPDATA4      = 1 <<  4
.equ VMP_WAIT_GPDATA5      = 1 <<  5
.equ VMP_WAIT_GPDATA6      = 1 <<  6
.equ VMP_WAIT_GPDATA7      = 1 <<  7
.equ VMP_WAIT_IRQOUT_SET   = 1 << 16
.equ VMP_WAIT_IRQOUT_CLEAR = 1 << 17
.equ VMP_WAIT_IRQIN_SET    = 1 << 18

.equ VMP_CTRL_ADDR_MASK = 0xfffffff8

/// @endcond
