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
* FILENAME: vmp_cl-stdlib.h
*
* DESCRIPTION: stdlib function definition
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

# ifndef __VMP_CL_STDLIB_H__
#  define __VMP_CL_STDLIB_H__

/*
 * ****************************************
 * some common stdlib function
 * ****************************************
 */

// abs
__attribute__((overloadable, always_inline))
inline unsigned int abs(int val)
{
  return __builtin_vmp_max(-val, val);
}


# endif // __VMP_CL_STDLIB_H__
