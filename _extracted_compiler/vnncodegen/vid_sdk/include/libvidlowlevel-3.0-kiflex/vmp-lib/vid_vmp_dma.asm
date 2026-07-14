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
 * FILENAME: vid_vmp_dma.asm
 *
 * DESCRIPTION: videantis v-MP DMA constants and DMA descriptors data section
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP DMA constants and DMA descriptors data section
 *
 * @details
 * This file defines the data section for v-MP DMA channel descriptors.
 * Placement and memory can be defined from the build system with
 * VMP_NUM_DMA_CHANNELS, VMP_DMA_DESCR_MEM, VMP_DMA_DESCR_ORG,
 * VMP_DMA_LEGACY_HIGH_ORG and VMP_DMA_LEGACY_HIGH
 *
 * @file vid_vmp_dma.asm
 */

/// @cond DOXYGEN_IGNORE_ASM


/* ========================================================================== *
 * dsection for DMA descriptors                                               *
 * ========================================================================== */
.if !defined(VMP_NUM_DMA_CHANNELS)
.equ VMP_NUM_DMA_CHANNELS = 16
.endif

// DMA channel descriptors base address
.if !defined(VMP_DMA_DESCR_ORG)
.equ VMP_DMA_DESCR_ORG = {0x1000 - 64}
.endif

// check if memory for dma descriptor dsection is defined (default dmem2)
.if !defined(VMP_DMA_DESCR_MEM)
.equ VMP_DMA_DESCR_MEM = 2
.endif

// translate dma descriptor dsection location
.if {VMP_DMA_DESCR_MEM == 1}
.equ DMA_DESCR_MEM = dmem
.elseif {VMP_DMA_DESCR_MEM == 3}
.equ DMA_DESCR_MEM = dmem3
.else
.equ DMA_DESCR_MEM = dmem2
.endif

// if no address is defined for placing memory for legacy dma transfer functions high part,
// place memory one word before dma descriptors
.if !defined(VMP_DMA_LEGACY_ORG)
.equ VMP_DMA_LEGACY_ORG = {VMP_DMA_DESCR_ORG - 1}
.endif

// check if a legacy high part is defined (default 0x00000100)
.if !defined(VMP_DMA_LEGACY_HIGH)
.equ VMP_DMA_LEGACY_HIGH = 0x00000100
.endif

// DMA channel descriptors
.dsection dma_descr, DMA_DESCR_MEM
.org VMP_DMA_DESCR_ORG
.align 64
.export _vmp_dma_descr
_vmp_dma_descr:
start:
    .loop dma = [0..VMP_NUM_DMA_CHANNELS-1]
start:
      .alloc(1) extByteaddr
      .alloc(1) xferFlags_intWordaddr_lengthInt_length
      .alloc(1) reserved_strideInt_count3d_count
      .alloc(1) stride3d_stride
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

    .equ DMA_PRIO_STD  = 0

    .equ DMA_CLEAR_CHANNEL = 1 << 7

    .equ DMA_MEMSET    = 1 << 5

    .equ DMA_READ      = 1 << 4
    .equ DMA_WRITE     = 0 << 4

    // DMA channel used for overlay manager
    .equ DMA_CH_OLM_NUM  = 15
    .equ DMA_CH_OLM_ADDR = {dma_descr.start + {DMA_CH_OLM_NUM * CHANNELSIZE}}
.endsection

// legacy DMA transfer
.dsection dma_legacy, DMA_DESCR_MEM
.org {VMP_DMA_LEGACY_ORG}, removeable
.export _vid_dma_legacy_high
_vid_dma_legacy_high:
    .alloc(1) high = {VMP_DMA_LEGACY_HIGH} // upper 32 bits of external 64-bit address for legacy transfer
.endsection

/// @endcond
