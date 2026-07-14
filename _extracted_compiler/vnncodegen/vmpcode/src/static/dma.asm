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
 * FILENAME:      dma.asm
 *
 * DESCRIPTION:   Constants and dsection for DMA control
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// DMA configuration registers
.equ BIU_BASE                   = 0x3B0
.equ BIU_DMA_DESCR_BASE         = BIU_BASE + 0
.equ BIU_DMA_CTRL               = BIU_BASE + 0x1
.equ BIU_DMA_STATUS             = BIU_BASE + 0x2
.equ BIU_BOOT_ADDRESS_L         = BIU_BASE + 0xE
.equ BIU_BOOT_ADDRESS_H         = BIU_BASE + 0xF


// DMA Descriptor
.equ VID_VMP_NUM_DMA_CHANNELS   = 12

.dsection dma_descr, dmem
.export DMA_PRIO_INT0
.export DMA_PRIO_INT1
.export DMA_PRIO_INT2
.export DMA_PRIO_INT3
.export DMA_PRIO_EXT0
.export DMA_PRIO_EXT1
.export DMA_PRIO_EXT2
.export DMA_PRIO_EXT3
.export DMA_PRIO_STD
.export DMA_PRIO_LOW
.export DMA_PRIO_HIGH

	.org   {0x0}
	.align 64
start:
	.loop	dma=[0..VID_VMP_NUM_DMA_CHANNELS-1]
start:
	  .alloc(1)     ext_byteaddr
	  .alloc(1)	xf__int_wordaddr__length_int__length
	  .alloc(1)	compr__stride_int__count_3d__count
	  .alloc(1)	stride_3d__stride
	.endloop

	.equ CHANNELSIZE = 4

	.equ DMA_PRIO_INT0 = 0b0000 << 8
	.equ DMA_PRIO_INT1 = 0b0001 << 8
	.equ DMA_PRIO_INT2 = 0b0010 << 8
	.equ DMA_PRIO_INT3 = 0b0011 << 8

	.equ DMA_PRIO_EXT0 = 0b0000 << 8
	.equ DMA_PRIO_EXT1 = 0b0100 << 8
	.equ DMA_PRIO_EXT2 = 0b1000 << 8
	.equ DMA_PRIO_EXT3 = 0b1100 << 8

	.equ DMA_PRIO_HIGH = DMA_PRIO_INT2
	.equ DMA_PRIO_STD  = DMA_PRIO_INT1
	.equ DMA_PRIO_LOW  = DMA_PRIO_INT0

	.equ DMA_MEMSET    = 1 << 5

	.equ DMA_READ      = 1 << 4
	.equ DMA_WRITE     = 0 << 4

.endsection
