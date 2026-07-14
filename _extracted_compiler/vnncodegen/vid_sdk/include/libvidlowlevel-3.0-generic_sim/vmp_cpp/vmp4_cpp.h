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
 * FILENAME: vmp4_cpp.h
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

#ifndef VMP4_CPP_H
#define VMP4_CPP_H

// data types
#include "vmp_cpp/vmp_cpp-types.h"

// functions
#include "vmp_cpp/vmp_cpp-functions.h"

// vmp-CL printf
#include "vmp_cpp/vmp_cpp-printf.h"

// connect to MISRA-C/C++
#include "vmp_cpp/vmp_cpp-misra.h"

// registers
#include "vmp_cpp/vmp_cpp_registers.h"

// DMA definitions and generic functions
#include "vmp_cpp/vmp_cpp_dma.h"

// DMA functions
#include "vmp_cpp/vmp_cpp_dma_func.h"

// DMA legacy functions
#include "vmp_cpp/vmp_cpp_dma_func_legacy.h"

// DMA endianness functions
#include "vmp_cpp/vmp_cpp_endianness_dma_func.h"

// EDMA definitions and functions
#include "vmp_cpp/vmp_cpp_edma.h"

// v-MP profiling functions
#include "vmp_cpp/vmp_cpp_profiling.h"

// v-MP prototypes for memory functions
#include "vmp_cpp/vmp_cpp-string.h"

#endif

