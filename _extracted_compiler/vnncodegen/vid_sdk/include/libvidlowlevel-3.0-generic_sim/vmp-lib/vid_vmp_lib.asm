/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2022 videantis GmbH
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
 * FILENAME: vid_vmp_lib.asm
 *
 * DESCRIPTION: videantis v-MP library main include file
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP lowlevel library main include file
 *
 * @details
 * This file includes all assembly files of the videantis v-MP lowlevel library.
 * This file must be included by the videantis linker vmpasm for
 * all user applications.
 *
 * @file vid_vmp_lib.asm
 */

/// @cond DOXYGEN_IGNORE_ASM

.include "vid_vmp_map.asm"
.include "vid_vmp_dma.asm"
.include "vid_vmp_edma.asm"
.include "vid_vmp_profiling.asm"
.include "vid_vmp_vmpcc.asm"
.include "vid_vmp_div.asm"
.include "vid_vmp_olm.asm"

/// @endcond
