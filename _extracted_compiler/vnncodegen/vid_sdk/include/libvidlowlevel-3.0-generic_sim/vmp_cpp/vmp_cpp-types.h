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
 * FILENAME: vmp_cpp-types.h
 *
 * DESCRIPTION: interface to videantis-C
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef VMP_CPP_TYPES_H
#define VMP_CPP_TYPES_H

// CL language compatibility
#define __LOCAL
#define __KERNEL

#define __global
#define __local
#define __constant const
#define __private
#define __kernel

#define global
#define local
#define constant const
#define kernel


# if ((__clang_major__ < 7) || defined(__videantis_lint__))
// LLVM 3
#  define __WORDADDRESS
#  define __BYTEADDRESS
# else
// LLVM 7
#  define __WORDADDRESS __wordaddress
#  define __BYTEADDRESS __byteaddress
# endif

# if (__clang_major__ < 7)
// LLVM 3
#  define SIZEOF_IN_WORDS(_type) sizeof(_type)
#  define SIZEOF_IN_BYTES(_type) (sizeof(_type) * 8)
# else
// LLVM 7
#  define SIZEOF_IN_WORDS(_type) (sizeof(_type) / 8)
#  define SIZEOF_IN_BYTES(_type) sizeof(_type)
# endif

/*
 * scalar types
 * for compatibility reasons most are defined, even if they are not supported
 */
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

/*
 * native vector data types
 */

/*
 * _64 - 1x64bit (for some instructions)
 * _32 - 2x32bit
 * _16 - 4x16bit
 * _8  - 8x8bit
 */
typedef long long1 __attribute__((ext_vector_type(1)));
typedef int int2 __attribute__((ext_vector_type(2)));
typedef short short4 __attribute__((ext_vector_type(4)));
typedef char char8 __attribute__((ext_vector_type(8)));

/*
 * _U64 - 1x64bit unsigned (for some instructions)
 * _U32 - 2x32bit unsigned
 * _U16 - 4x16bit unsigned
 * _U8  - 8x8bit unsigned
 */
typedef unsigned long ulong1 __attribute__((ext_vector_type(1)));
typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
typedef unsigned short ushort4 __attribute__((ext_vector_type(4)));
typedef unsigned char uchar8 __attribute__((ext_vector_type(8)));


/*
 * double width data types (for V_MAC)
 */
typedef long long2 __attribute__((ext_vector_type(2)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef short short8 __attribute__((ext_vector_type(8)));

typedef unsigned long ulong2 __attribute__((ext_vector_type(2)));
typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
typedef unsigned short ushort8 __attribute__((ext_vector_type(8)));

/*
 * double width data types (for V_SADALIGN)
 */
typedef char char16 __attribute__((ext_vector_type(16)));
typedef unsigned char uchar16 __attribute__((ext_vector_type(16)));

/// data type used for v-MP register accesses
using addr_t = uint;
/// data type used for v-MP register accesses
using reg32 = volatile uint;

#endif // vmp_cpp_types_h
