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
* FILENAME: vmp_cl.h
*
* DESCRIPTION: include interfaces for videantis-C on v-MP 3.x
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

# ifndef __VMP_CL_H__
#  define __VMP_CL_H__

// data types
#  include "vmp_cl-types.h"

// attributes
#  include "vmp_cl-attributes.h"

// functions
#  include "vmp_cl-functions.h"

// some stdlib functions as
#  include "vmp_cl-stdlib.h"

// vmp-CL DMA
#  include "vmp_cl-dma.h"

// vmp-CL printf
#  include "vmp_cl-printf.h"

// connect to MISRA-C/C++
#  include "vmp_cl-misra.h"

// memcpy, memcmp, memset
#  include "vmp_cl-string.h"

# endif // __vmp_cl_h__
