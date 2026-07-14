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
 * FILENAME: vid_vmp_edma.asm
 *
 * DESCRIPTION: definitions for external DMA transfer functions
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// EDMA channel descriptors memory
.if !defined(VMP_EDMA_DESCR_MEM)
.equ VMP_EDMA_DESCR_MEM = 2
.endif

// translate dma descriptor dsection location
.if {VMP_EDMA_DESCR_MEM == 1}
.equ EDMA_DESCR_MEM = dmem
.elseif {VMP_EDMA_DESCR_MEM == 3}
.equ EDMA_DESCR_MEM = dmem3
.else
.equ EDMA_DESCR_MEM = dmem2
.endif

// EDMA channel descriptors base address
// if not defined place before DMA descriptors and memory for legacy dma transfer functions high part
.if !defined(VMP_EDMA_DESCR_ORG)
.equ VMP_EDMA_DESCR_ORG = {VMP_DMA_DESCR_ORG - 7}
.endif

// EDMA channel descriptor, external address and channel status
.dsection edma_descr, EDMA_DESCR_MEM
.org {VMP_EDMA_DESCR_ORG}, removeable
.export _vid_edma_descr
.export _vid_edma_extDescrAddr
.export _vid_edma_chStatus
_vid_edma_descr:
    .alloc(1) tgtAddr
    .alloc(1) srcAddr
    .alloc(1) flags_count_lengthHi_lengthLo
    .alloc(1) tgtStride_srcStride
_vid_edma_extDescrAddr:
    .alloc(1) extDescrAddr
_vid_edma_chStatus:
    .alloc(1) chStatus
.endsection
