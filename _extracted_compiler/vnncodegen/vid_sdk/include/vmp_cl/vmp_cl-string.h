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
* FILENAME: vmp_cl-string.h
*
* DESCRIPTION: prototypes for memory functions
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

# ifndef __VMP_CL_STRING_H__
#  define __VMP_CL_STRING_H__

// attributes
#  include "vmp_cl-attributes.h"

// some stddef functions as
#  include "vmp_cl-stddef.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cl/vid_videantis-C.hpp"
#endif

// HINT we mark addresses as __byteaddress as a remainder that there
// will be a byte address.

// add prototypes for memory functions
int memcmp(__byteaddress __local const void *str1,
           __byteaddress __local const void *str2, size_t n);
__local void *memcpy(__byteaddress __local void *dest,
                     __byteaddress __local const void *src, size_t n);
__local void *memset(__byteaddress __local void *str, int c, size_t n);

# endif // __VMP_CL_STRING_H__
