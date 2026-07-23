/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2021 videantis GmbH
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
 * FILENAME:      vid_vmp_boot_loader_simple.asm
 *
 * DESCRIPTION:   videantis v-MP bootloader
 *                Simplified version: no data initialization, no LLVM setup
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// CYCLE_COUNT_IGNORE_WAIT - configure the clock cycle counter
//
// If defined and nonzero: Clock cycles will be counted only when MP_STATE=RUN
//                         (i.e. any WAIT cycles will be ignored)
// If undefined or zero:   Count all cycles including any WAIT cycles
.equ CYCLE_COUNT_IGNORE_WAIT = 1

/* ========================================================================== *
 *         Boot code segment                                                  *
 * ========================================================================== */
.csection init
	.equ VID_IMEM_START_ADDR_AFTER_INITIAL_CODE	= 0x00000030
	.equ VID_IMEM_SIZE_RESIDENT_CODE                = VIDASM_imem_LAST_NON_OVERLAY_ADDRESS + 1
	.equ VID_IMEM_LENGTH_AFTER_INITIAL_CODE         = VID_IMEM_SIZE_RESIDENT_CODE - VID_IMEM_START_ADDR_AFTER_INITIAL_CODE

	.org 0x0000

vmp_bootloader:
	MVI	BIU_DMA_DESCR_BASE, #dma_descr.start
	/* load resident code into intruction memory using DMA channel 0 */
	// instruction memory starts at 0xf000
	// do not load initial 48 words already loaded by HW after booting
	MVFIRI	SFIR0, #dma_descr.start		// descriptor address for channel 0
	MV	STORE_HIGH, BIU_BOOT_ADDRESS_H
	MV	R0, BIU_BOOT_ADDRESS_L
	MVI	R1,#VID_IMEM_START_ADDR_AFTER_INITIAL_CODE*8
	ADD	(SFIR0)+, R0, R1
	MVIL	STORE_HIGH, #VID_IMEM_START_ADDR_AFTER_INITIAL_CODE+0xf000
	MVIL	(SFIR0)+, #VID_IMEM_LENGTH_AFTER_INITIAL_CODE
	MVI	(SFIR0)+, #0

	.dependency dmem2, 0
	MVI     BIU_DMA_CTRL, #{0 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD }

	/* wait for all transfers to be finished */
	WAIT    #0xffff

.if defined(STANDALONE_HANDSHAKE) && {STANDALONE_HANDSHAKE != 0}
	MV	GPDATA1, ONE
boot_sync:
	MV	SR0, GPDATA1
	ORCS	SR0, ZERO, SR0
	BSSR	boot_sync, #COND_NZ


	// set up timer to count clock cycles in GPDATA1
	MVI	TIMER1, #0
	MV	GPDATA1, TIMER1
.if defined(CYCLE_COUNT_IGNORE_WAIT) && {CYCLE_COUNT_IGNORE_WAIT != 0}
	MVI	TCTRL1, #0b0001	// count up, count only when MP_STATE=RUN
.else
	MVI	TCTRL1, #0b1001	// count up, count independent from MP_STATE
.endif

.endif

	/* get mp core ID from boot address */
	MV	SR0, BIU_BOOT_ADDRESS_L		// 0x60n00000
	MV	STORE_HIGH, BIU_BOOT_ADDRESS_H
	MV	boot.boot_address, SR0

	// Continue at main after booloader finished; never return from there
	JLR	ZERO, #_main
.endsection

.dsection boot
	.alloc boot_address
.endsection


.end
