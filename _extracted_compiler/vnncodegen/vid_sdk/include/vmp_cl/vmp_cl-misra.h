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
* FILENAME: vmp_cl-misra.h
*
* DESCRIPTION: interface to functions required for MISRA-C/C++
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/


# ifndef __VMP_CL_MISRA_H__
#  define __VMP_CL_MISRA_H__

#ifndef __videantis_lint__
/*
 * These macros are required for MISRA-C/C++ checks on OpenCL-C code
 */
#  define set_long1 (long1)
#  define set_ulong1 (ulong1)
#  define set_int2 (int2)
#  define set_uint2 (uint2)
#  define set_short2 (short2)
#  define set_ushort2 (ushort2)
#  define set_char2 (char2)
#  define set_uchar2 (uchar2)
#  define set_char4 (char4)
#  define set_char4 (char4)
#  define set_uchar4 (uchar4)
#  define set_short4 (short4)
#  define set_ushort4 (ushort4)
#  define set_char8 (char8)
#  define set_uchar8 (uchar8)
#  define set_long2 (long2)
#  define set_ulong2 (ulong2)
#  define set_int4 (int4)
#  define set_uint4 (uint4)
#  define set_short8 (short8)
#  define set_ushort8 (ushort8)
#  define set_char16 (char16)
#  define set_uchar16 (uchar16)

#endif  // __videantis_lint__

# endif // __VMP_CL_MISRA_H__

