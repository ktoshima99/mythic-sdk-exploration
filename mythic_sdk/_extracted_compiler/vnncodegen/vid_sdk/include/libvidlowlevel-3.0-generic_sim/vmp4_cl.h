/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2024 videantis GmbH
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
 * FILENAME: vmp4_cl.h
 *
 * DESCRIPTION: include interfaces for videantis-C on v-MP 4.x
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @file
 * @brief include interfaces for videantis-C on v-MP 4.x
 *
 * This file includes all needed interfaces for videantis-C on v-MP 4.x
 *
 */

#ifndef __VMP4_CL_H__
#define __VMP4_CL_H__

// data types
#include "vmp_cl/vmp_cl-types.h"

// functions
#include "vmp_cl/vmp_cl-functions.h"

// vmp-CL printf
#include "vmp_cl/vmp_cl-printf.h"

// connect to MISRA-C/C++
#include "vmp_cl/vmp_cl-misra.h"

// DMA definitions and generic functions
#include "vmp-lib/vid_vmp_dma.h"

// DMA functions
#include "vmp-lib/vid_vmp_dma_func.h"

// DMA legacy functions
#include "vmp-lib/vid_vmp_dma_func_legacy.h"

// DMA endianness functions
#include "vmp-lib/vid_vmp_endianness_dma_func.h"

// EDMA definitions and functions
#include "vmp-lib/vid_vmp_edma.h"

// v-MP profiling functions
#include "vmp-lib/vid_vmp_profiling.h"

// v-MP prototypes for memory functions
#include "vmp_cl/vmp_cl-string.h"

#endif

