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
 * FILENAME: vmp_cl-functions.h
 *
 * DESCRIPTION: interface to videantis-C
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

# ifndef __VMP_CL_FUNCTIONS_H__
#  define __VMP_CL_FUNCTIONS_H__

// attributes
#include "vmp_cl-attributes.h"

// connect to MISRA-C/C++
#include "vmp_cl-misra.h"

// vector data types
#include "vmp_cl/vmp_cl-types.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cl/vid_videantis-C.hpp"

#else
/*
 * ****************************************
 * convert_TYPE() conversion
 * ****************************************
 */

/*
 * OpenCL: The number of elements in the source and destination
 * vectors must match.
 *
 * LLVM: The input vector and the output vector type must have the
 * same number of elements.
 *
 * __builtin_convertvector is used to express generic vector
 * type-conversion operations.
 */

#define convert_long1(_op)  __builtin_convertvector(_op, long1)
#define convert_int2(_op)   __builtin_convertvector(_op, int2)
#define convert_short4(_op) __builtin_convertvector(_op, short4)
#define convert_char8(_op)  __builtin_convertvector(_op, char8)

#define convert_ulong1(_op)  __builtin_convertvector(_op, ulong1)
#define convert_uint2(_op)   __builtin_convertvector(_op, uint2)
#define convert_ushort4(_op) __builtin_convertvector(_op, ushort4)
#define convert_uchar8(_op)  __builtin_convertvector(_op, uchar8)

/*
 * convertions for extended vector support
 */
#define convert_long2(_op)  __builtin_convertvector(_op, long2)
#define convert_int4(_op)   __builtin_convertvector(_op, int4)
#define convert_short8(_op) __builtin_convertvector(_op, short8)

#define convert_ulong2(_op)  __builtin_convertvector(_op, ulong2)
#define convert_uint4(_op)   __builtin_convertvector(_op, uint4)
#define convert_ushort8(_op) __builtin_convertvector(_op, ushort8)

#define builtin_shuffle8(_a, _b, _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7) __builtin_shufflevector(_a, _b, _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7)
#define builtin_shuffle4(_a, _b, _s0, _s1, _s2, _s3) __builtin_shufflevector(_a, _b, _s0, _s1, _s2, _s3)
#define builtin_shuffle2(_a, _b, _s0, _s1) __builtin_shufflevector(_a, _b, _s0, _s1)

#endif  // __videantis_lint__

/*
 * ****************************************
 * as_TYPE() conversion
 * ****************************************
 */

/*
 * native vector data-types
 */

// char8
__attribute__((overloadable, always_inline))
inline char8 as_char8 (char8 v0)
{
  return __builtin_vmp_bitcast_8(v0);
}

__attribute__((overloadable, always_inline))
inline char8 as_char8 (uchar8 v0)
{
  return __builtin_vmp_bitcast_8(v0);
}

__attribute__((overloadable, always_inline))
inline char8 as_char8 (short4 v0)
{
  return __builtin_vmp_bitcast_8(v0);
}

__attribute__((overloadable, always_inline))
inline char8 as_char8 (ushort4 v0)
{
  return __builtin_vmp_bitcast_8(v0);
}

__attribute__((overloadable, always_inline))
inline char8 as_char8 (int2 v0)
{
  return __builtin_vmp_bitcast_8(v0);
}

__attribute__((overloadable, always_inline))
inline char8 as_char8 (uint2 v0)
{
  return __builtin_vmp_bitcast_8(v0);
}

__attribute__((overloadable, always_inline))
inline char8 as_char8 (long1 v0)
{
  return __builtin_vmp_bitcast_8(v0);
}

__attribute__((overloadable, always_inline))
inline char8 as_char8 (ulong1 v0)
{
  return __builtin_vmp_bitcast_8(v0);
}

// uchar8
__attribute__((overloadable, always_inline))
inline uchar8 as_uchar8 (char8 v0)
{
  return convert_uchar8(__builtin_vmp_bitcast_8(v0));
}

__attribute__((overloadable, always_inline))
inline uchar8 as_uchar8 (uchar8 v0)
{
  return convert_uchar8(__builtin_vmp_bitcast_8(v0));
}

__attribute__((overloadable, always_inline))
inline uchar8 as_uchar8 (short4 v0)
{
  return convert_uchar8(__builtin_vmp_bitcast_8(v0));
}

__attribute__((overloadable, always_inline))
inline uchar8 as_uchar8 (ushort4 v0)
{
  return convert_uchar8(__builtin_vmp_bitcast_8(v0));
}

__attribute__((overloadable, always_inline))
inline uchar8 as_uchar8 (int2 v0)
{
  return convert_uchar8(__builtin_vmp_bitcast_8(v0));
}

__attribute__((overloadable, always_inline))
inline uchar8 as_uchar8 (uint2 v0)
{
  return convert_uchar8(__builtin_vmp_bitcast_8(v0));
}

__attribute__((overloadable, always_inline))
inline uchar8 as_uchar8 (long1 v0)
{
  return convert_uchar8(__builtin_vmp_bitcast_8(v0));
}

__attribute__((overloadable, always_inline))
inline uchar8 as_uchar8 (ulong1 v0)
{
  return convert_uchar8(__builtin_vmp_bitcast_8(v0));
}

// short4
__attribute__((overloadable, always_inline))
inline short4 as_short4 (char8 v0)
{
  return __builtin_vmp_bitcast_16(v0);
}

__attribute__((overloadable, always_inline))
inline short4 as_short4 (uchar8 v0)
{
  return __builtin_vmp_bitcast_16(v0);
}

__attribute__((overloadable, always_inline))
inline short4 as_short4 (short4 v0)
{
  return __builtin_vmp_bitcast_16(v0);
}

__attribute__((overloadable, always_inline))
inline short4 as_short4 (ushort4 v0)
{
  return __builtin_vmp_bitcast_16(v0);
}

__attribute__((overloadable, always_inline))
inline short4 as_short4 (int2 v0)
{
  return __builtin_vmp_bitcast_16(v0);
}

__attribute__((overloadable, always_inline))
inline short4 as_short4 (uint2 v0)
{
  return __builtin_vmp_bitcast_16(v0);
}

__attribute__((overloadable, always_inline))
inline short4 as_short4 (long1 v0)
{
  return __builtin_vmp_bitcast_16(v0);
}

__attribute__((overloadable, always_inline))
inline short4 as_short4 (ulong1 v0)
{
  return __builtin_vmp_bitcast_16(v0);
}

// ushort4
__attribute__((overloadable, always_inline))
inline ushort4 as_ushort4 (char8 v0)
{
  return convert_ushort4(__builtin_vmp_bitcast_16(v0));
}

__attribute__((overloadable, always_inline))
inline ushort4 as_ushort4 (uchar8 v0)
{
  return convert_ushort4(__builtin_vmp_bitcast_16(v0));
}

__attribute__((overloadable, always_inline))
inline ushort4 as_ushort4 (short4 v0)
{
  return convert_ushort4(__builtin_vmp_bitcast_16(v0));
}

__attribute__((overloadable, always_inline))
inline ushort4 as_ushort4 (ushort4 v0)
{
  return convert_ushort4(__builtin_vmp_bitcast_16(v0));
}

__attribute__((overloadable, always_inline))
inline ushort4 as_ushort4 (int2 v0)
{
  return convert_ushort4(__builtin_vmp_bitcast_16(v0));
}

__attribute__((overloadable, always_inline))
inline ushort4 as_ushort4 (uint2 v0)
{
  return convert_ushort4(__builtin_vmp_bitcast_16(v0));
}

__attribute__((overloadable, always_inline))
inline ushort4 as_ushort4 (long1 v0)
{
  return convert_ushort4(__builtin_vmp_bitcast_16(v0));
}

__attribute__((overloadable, always_inline))
inline ushort4 as_ushort4 (ulong1 v0)
{
  return convert_ushort4(__builtin_vmp_bitcast_16(v0));
}

// int2
__attribute__((overloadable, always_inline))
inline int2 as_int2 (char8 v0)
{
  return __builtin_vmp_bitcast_32(v0);
}

__attribute__((overloadable, always_inline))
inline int2 as_int2 (uchar8 v0)
{
  return __builtin_vmp_bitcast_32(v0);
}

__attribute__((overloadable, always_inline))
inline int2 as_int2 (short4 v0)
{
  return __builtin_vmp_bitcast_32(v0);
}

__attribute__((overloadable, always_inline))
inline int2 as_int2 (ushort4 v0)
{
  return __builtin_vmp_bitcast_32(v0);
}

__attribute__((overloadable, always_inline))
inline int2 as_int2 (int2 v0)
{
  return __builtin_vmp_bitcast_32(v0);
}

__attribute__((overloadable, always_inline))
inline int2 as_int2 (uint2 v0)
{
  return __builtin_vmp_bitcast_32(v0);
}

__attribute__((overloadable, always_inline))
inline int2 as_int2 (long1 v0)
{
  return __builtin_vmp_bitcast_32(v0);
}

__attribute__((overloadable, always_inline))
inline int2 as_int2 (ulong1 v0)
{
  return __builtin_vmp_bitcast_32(v0);
}

// uint2
__attribute__((overloadable, always_inline))
inline uint2 as_uint2 (char8 v0)
{
  return convert_uint2(__builtin_vmp_bitcast_32(v0));
}

__attribute__((overloadable, always_inline))
inline uint2 as_uint2 (uchar8 v0)
{
  return convert_uint2(__builtin_vmp_bitcast_32(v0));
}

__attribute__((overloadable, always_inline))
inline uint2 as_uint2 (short4 v0)
{
  return convert_uint2(__builtin_vmp_bitcast_32(v0));
}

__attribute__((overloadable, always_inline))
inline uint2 as_uint2 (ushort4 v0)
{
  return convert_uint2(__builtin_vmp_bitcast_32(v0));
}

__attribute__((overloadable, always_inline))
inline uint2 as_uint2 (int2 v0)
{
  return convert_uint2(__builtin_vmp_bitcast_32(v0));
}

__attribute__((overloadable, always_inline))
inline uint2 as_uint2 (uint2 v0)
{
  return convert_uint2(__builtin_vmp_bitcast_32(v0));
}

__attribute__((overloadable, always_inline))
inline uint2 as_uint2 (long1 v0)
{
  return convert_uint2(__builtin_vmp_bitcast_32(v0));
}

__attribute__((overloadable, always_inline))
inline uint2 as_uint2 (ulong1 v0)
{
  return convert_uint2(__builtin_vmp_bitcast_32(v0));
}


// long1
__attribute__((overloadable, always_inline))
inline long1 as_long1 (char8 v0)
{
  return __builtin_vmp_bitcast_64(v0);
}

__attribute__((overloadable, always_inline))
inline long1 as_long1 (uchar8 v0)
{
  return __builtin_vmp_bitcast_64(v0);
}

__attribute__((overloadable, always_inline))
inline long1 as_long1 (short4 v0)
{
  return __builtin_vmp_bitcast_64(v0);
}

__attribute__((overloadable, always_inline))
inline long1 as_long1 (ushort4 v0)
{
  return __builtin_vmp_bitcast_64(v0);
}

__attribute__((overloadable, always_inline))
inline long1 as_long1 (int2 v0)
{
  return __builtin_vmp_bitcast_64(v0);
}

__attribute__((overloadable, always_inline))
inline long1 as_long1 (uint2 v0)
{
  return __builtin_vmp_bitcast_64(v0);
}

__attribute__((overloadable, always_inline))
inline long1 as_long1 (long1 v0)
{
  return __builtin_vmp_bitcast_64(v0);
}

__attribute__((overloadable, always_inline))
inline long1 as_long1 (ulong1 v0)
{
  return __builtin_vmp_bitcast_64(v0);
}

// ulong1
__attribute__((overloadable, always_inline))
inline ulong1 as_ulong1 (char8 v0)
{
  return convert_ulong1(__builtin_vmp_bitcast_64(v0));
}

__attribute__((overloadable, always_inline))
inline ulong1 as_ulong1 (uchar8 v0)
{
  return convert_ulong1(__builtin_vmp_bitcast_64(v0));
}

__attribute__((overloadable, always_inline))
inline ulong1 as_ulong1 (short4 v0)
{
  return convert_ulong1(__builtin_vmp_bitcast_64(v0));
}

__attribute__((overloadable, always_inline))
inline ulong1 as_ulong1 (ushort4 v0)
{
  return convert_ulong1(__builtin_vmp_bitcast_64(v0));
}

__attribute__((overloadable, always_inline))
inline ulong1 as_ulong1 (int2 v0)
{
  return convert_ulong1(__builtin_vmp_bitcast_64(v0));
}

__attribute__((overloadable, always_inline))
inline ulong1 as_ulong1 (uint2 v0)
{
  return convert_ulong1(__builtin_vmp_bitcast_64(v0));
}

__attribute__((overloadable, always_inline))
inline ulong1 as_ulong1 (long1 v0)
{
  return convert_ulong1(__builtin_vmp_bitcast_64(v0));
}

__attribute__((overloadable, always_inline))
inline ulong1 as_ulong1 (ulong1 v0)
{
  return convert_ulong1(__builtin_vmp_bitcast_64(v0));
}

/*
 * non-native vector data-types
 */

// short8
__attribute__((overloadable, always_inline))
inline short8 as_short8 (short8 v0)
{
  return __builtin_vmp_bitcast_16x2(v0);
}

__attribute__((overloadable, always_inline))
inline short8 as_short8 (ushort8 v0)
{
  return __builtin_vmp_bitcast_16x2(v0);
}

__attribute__((overloadable, always_inline))
inline short8 as_short8 (int4 v0)
{
  return __builtin_vmp_bitcast_16x2(v0);
}

__attribute__((overloadable, always_inline))
inline short8 as_short8 (uint4 v0)
{
  return __builtin_vmp_bitcast_16x2(v0);
}

__attribute__((overloadable, always_inline))
inline short8 as_short8 (long2 v0)
{
  return __builtin_vmp_bitcast_16x2(v0);
}

__attribute__((overloadable, always_inline))
inline short8 as_short8 (ulong2 v0)
{
  return __builtin_vmp_bitcast_16x2(v0);
}

// ushort8
__attribute__((overloadable, always_inline))
inline ushort8 as_ushort8 (short8 v0)
{
  return convert_ushort8(v0);
}

__attribute__((overloadable, always_inline))
inline ushort8 as_ushort8 (ushort8 v0)
{
  return convert_ushort8(__builtin_vmp_bitcast_16x2(v0));
}

__attribute__((overloadable, always_inline))
inline ushort8 as_ushort8 (int4 v0)
{
  return convert_ushort8(__builtin_vmp_bitcast_16x2(v0));
}

__attribute__((overloadable, always_inline))
inline ushort8 as_ushort8 (uint4 v0)
{
  return convert_ushort8(__builtin_vmp_bitcast_16x2(v0));
}

__attribute__((overloadable, always_inline))
inline ushort8 as_ushort8 (long2 v0)
{
  return convert_ushort8(__builtin_vmp_bitcast_16x2(v0));
}

__attribute__((overloadable, always_inline))
inline ushort8 as_ushort8 (ulong2 v0)
{
  return convert_ushort8(__builtin_vmp_bitcast_16x2(v0));
}


// int4
__attribute__((overloadable, always_inline))
inline int4 as_int4 (short8 v0)
{
  return __builtin_vmp_bitcast_32x2(v0);
}

__attribute__((overloadable, always_inline))
inline int4 as_int4 (ushort8 v0)
{
  return __builtin_vmp_bitcast_32x2(v0);
}

__attribute__((overloadable, always_inline))
inline int4 as_int4 (int4 v0)
{
  return __builtin_vmp_bitcast_32x2(v0);
}

__attribute__((overloadable, always_inline))
inline int4 as_int4 (uint4 v0)
{
  return __builtin_vmp_bitcast_32x2(v0);
}

__attribute__((overloadable, always_inline))
inline int4 as_int4 (long2 v0)
{
  return __builtin_vmp_bitcast_32x2(v0);
}

__attribute__((overloadable, always_inline))
inline int4 as_int4 (ulong2 v0)
{
  return __builtin_vmp_bitcast_32x2(v0);
}

// uint4
__attribute__((overloadable, always_inline))
inline uint4 as_uint4 (short8 v0)
{
  return convert_uint4(__builtin_vmp_bitcast_32x2(v0));
}

__attribute__((overloadable, always_inline))
inline uint4 as_uint4 (ushort8 v0)
{
  return convert_uint4(__builtin_vmp_bitcast_32x2(v0));
}

__attribute__((overloadable, always_inline))
inline uint4 as_uint4 (int4 v0)
{
  return convert_uint4(__builtin_vmp_bitcast_32x2(v0));
}

__attribute__((overloadable, always_inline))
inline uint4 as_uint4 (uint4 v0)
{
  return convert_uint4(__builtin_vmp_bitcast_32x2(v0));
}

__attribute__((overloadable, always_inline))
inline uint4 as_uint4 (long2 v0)
{
  return convert_uint4(__builtin_vmp_bitcast_32x2(v0));
}

__attribute__((overloadable, always_inline))
inline uint4 as_uint4 (ulong2 v0)
{
  return convert_uint4(__builtin_vmp_bitcast_32x2(v0));
}


// long2
__attribute__((overloadable, always_inline))
inline long2 as_long2 (short8 v0)
{
  return __builtin_vmp_bitcast_64x2(v0);
}

__attribute__((overloadable, always_inline))
inline long2 as_long2 (ushort8 v0)
{
  return __builtin_vmp_bitcast_64x2(v0);
}

__attribute__((overloadable, always_inline))
inline long2 as_long2 (int4 v0)
{
  return __builtin_vmp_bitcast_64x2(v0);
}

__attribute__((overloadable, always_inline))
inline long2 as_long2 (uint4 v0)
{
  return __builtin_vmp_bitcast_64x2(v0);
}

__attribute__((overloadable, always_inline))
inline long2 as_long2 (long2 v0)
{
  return __builtin_vmp_bitcast_64x2(v0);
}

__attribute__((overloadable, always_inline))
inline long2 as_long2 (ulong2 v0)
{
  return __builtin_vmp_bitcast_64x2(v0);
}

// ulong2
__attribute__((overloadable, always_inline))
inline ulong2 as_ulong2 (short8 v0)
{
  return convert_ulong2(__builtin_vmp_bitcast_64x2(v0));
}

__attribute__((overloadable, always_inline))
inline ulong2 as_ulong2 (ushort8 v0)
{
  return convert_ulong2(__builtin_vmp_bitcast_64x2(v0));
}

__attribute__((overloadable, always_inline))
inline ulong2 as_ulong2 (int4 v0)
{
  return convert_ulong2(__builtin_vmp_bitcast_64x2(v0));
}

__attribute__((overloadable, always_inline))
inline ulong2 as_ulong2 (uint4 v0)
{
  return convert_ulong2(__builtin_vmp_bitcast_64x2(v0));
}

__attribute__((overloadable, always_inline))
inline ulong2 as_ulong2 (long2 v0)
{
  return convert_ulong2(__builtin_vmp_bitcast_64x2(v0));
}

__attribute__((overloadable, always_inline))
inline ulong2 as_ulong2 (ulong2 v0)
{
  return convert_ulong2(__builtin_vmp_bitcast_64x2(v0));
}


/*
 * ****************************************
 * 6.13.3 integer built-in functions
 * ****************************************
 */

// abs (gentype x)
__attribute__((overloadable, always_inline))
inline uchar8 vabs (char8 x)
{
  return convert_uchar8(__builtin_vmp_vabsadd_8(x, set_char8(0,0,0,0,0,0,0,0)));
}

__attribute__((overloadable, always_inline))
inline uchar8 vabs (uchar8 x)
{
  return x;
}

__attribute__((overloadable, always_inline))
inline ushort4 vabs (short4 x)
{
  return convert_ushort4(__builtin_vmp_vabsadd_16(x, set_short4(0,0,0,0)));
}

__attribute__((overloadable, always_inline))
inline ushort4 vabs (ushort4 x)
{
  return x;
}

__attribute__((overloadable, always_inline))
inline uint2 vabs (int2 x)
{
  return convert_uint2(__builtin_vmp_vabsadd_32(x, set_int2(0,0)));
}

__attribute__((overloadable, always_inline))
inline uint2 vabs (uint2 x)
{
  return x;
}


// ugentype abs_diff (gentype x, gentype y)
__attribute__((overloadable, always_inline))
inline uchar8 abs_diff (char8 x, char8 y)
{
  return convert_uchar8(__builtin_vmp_vmax_8(x, y) - __builtin_vmp_vmin_8(x, y));
}

__attribute__((overloadable, always_inline))
inline uchar8 abs_diff (uchar8 x, uchar8 y)
{
  return __builtin_vmp_vmax_u8(x, y) - __builtin_vmp_vmin_u8(x, y);
}

__attribute__((overloadable, always_inline))
inline ushort4 abs_diff (short4 x, short4 y)
{
  return convert_ushort4(__builtin_vmp_vmax_16(x, y) - __builtin_vmp_vmin_16(x, y));
}

__attribute__((overloadable, always_inline))
inline ushort4 abs_diff (ushort4 x, ushort4 y)
{
  return __builtin_vmp_vmax_u16(x, y) - __builtin_vmp_vmin_u16(x, y);
}

__attribute__((overloadable, always_inline))
inline uint2 abs_diff (int2 x, int2 y)
{
  return convert_uint2(__builtin_vmp_vmax_32(x, y) - __builtin_vmp_vmin_32(x, y));
}

__attribute__((overloadable, always_inline))
inline uint2 abs_diff (uint2 x, uint2 y)
{
  return __builtin_vmp_vmax_u32(x, y) - __builtin_vmp_vmin_u32(x, y);
}


// gentype add_sat (gentype x, gentype y)
__attribute__((overloadable, always_inline))
inline char8 add_sat (char8 x, char8 y)
{
  return __builtin_vmp_vadd_8s(x, y);
}

__attribute__((overloadable, always_inline))
inline uchar8 add_sat (uchar8 x, uchar8 y)
{
  return __builtin_vmp_vadd_u8s(x, y);
}

__attribute__((overloadable, always_inline))
inline short4 add_sat (short4 x, short4 y)
{
  return __builtin_vmp_vadd_16s(x, y);
}

__attribute__((overloadable, always_inline))
inline ushort4 add_sat (ushort4 x, ushort4 y)
{
  return __builtin_vmp_vadd_u16s(x, y);
}

__attribute__((overloadable, always_inline))
inline int2 add_sat (int2 x, int2 y)
{
  return __builtin_vmp_vadd_32s(x, y);
}

__attribute__((overloadable, always_inline))
inline uint2 add_sat (uint2 x, uint2 y)
{
  return __builtin_vmp_vadd_u32s(x, y);
}

// gentype hadd (gentype x, gentype y)
__attribute__((overloadable, always_inline))
inline char8 hadd (char8 x, char8 y)
{
  return __builtin_vmp_vavgrc_8(x, y, 0);
}

__attribute__((overloadable, always_inline))
inline uchar8 hadd (uchar8 x, uchar8 y)
{
  return __builtin_vmp_vavgrc_u8(x, y, 0);
}

__attribute__((overloadable, always_inline))
inline short4 hadd (short4 x, short4 y)
{
  return __builtin_vmp_vavgrc_16(x, y, 0);
}

__attribute__((overloadable, always_inline))
inline ushort4 hadd (ushort4 x, ushort4 y)
{
  return __builtin_vmp_vavgrc_u16(x, y, 0);
}

__attribute__((overloadable, always_inline))
inline int2 hadd (int2 x, int2 y)
{
  return __builtin_vmp_vavgrc_32(x, y, 0);
}

__attribute__((overloadable, always_inline))
inline uint2 hadd (uint2 x, uint2 y)
{
  return __builtin_vmp_vavgrc_u32(x, y, 0);
}

// gentype rhadd (gentype x, gentype y)
__attribute__((overloadable, always_inline))
inline char8 rhadd (char8 x, char8 y)
{
  return __builtin_vmp_vavg_8(x, y);
}

__attribute__((overloadable, always_inline))
inline uchar8 rhadd (uchar8 x, uchar8 y)
{
  return __builtin_vmp_vavg_u8(x, y);
}

__attribute__((overloadable, always_inline))
inline short4 rhadd (short4 x, short4 y)
{
  return __builtin_vmp_vavg_16(x, y);
}

__attribute__((overloadable, always_inline))
inline ushort4 rhadd (ushort4 x, ushort4 y)
{
  return __builtin_vmp_vavg_u16(x, y);
}

__attribute__((overloadable, always_inline))
inline int2 rhadd (int2 x, int2 y)
{
  return __builtin_vmp_vavg_32(x, y);
}

__attribute__((overloadable, always_inline))
inline uint2 rhadd (uint2 x, uint2 y)
{
  return __builtin_vmp_vavg_u32(x, y);
}

// gentype clamp (gentype x, sgentype minval, sgentype maxval)
__attribute__((overloadable, always_inline))
inline char8 clamp (char8 x, char8 minval, char8 maxval)
{
  return __builtin_vmp_vmin_8(__builtin_vmp_vmax_8(x, minval), maxval);
}

__attribute__((overloadable, always_inline))
inline short4 clamp (short4 x, short4 minval, short4 maxval)
{
  return __builtin_vmp_vmin_16(__builtin_vmp_vmax_16(x, minval), maxval);
}

__attribute__((overloadable, always_inline))
inline int2 clamp (int2 x, int2 minval, int2 maxval)
{
  return __builtin_vmp_vmin_32(__builtin_vmp_vmax_32(x, minval), maxval);
}

__attribute__((overloadable, always_inline))
inline uchar8 clamp (uchar8 x, uchar8 minval, uchar8 maxval)
{
  return __builtin_vmp_vmin_u8(__builtin_vmp_vmax_u8(x, minval), maxval);
}

__attribute__((overloadable, always_inline))
inline ushort4 clamp (ushort4 x, ushort4 minval, ushort4 maxval)
{
  return __builtin_vmp_vmin_u16(__builtin_vmp_vmax_u16(x, minval), maxval);
}

__attribute__((overloadable, always_inline))
inline uint2 clamp (uint2 x, uint2 minval, uint2 maxval)
{
  return __builtin_vmp_vmin_u32(__builtin_vmp_vmax_u32(x, minval), maxval);
}

// gentype clz (gentype x)
__attribute__((overloadable, always_inline))
inline char8 clz (char8 x)
{
  return as_char8(__builtin_vmp_clz_8(as_uchar8(x)));
}

__attribute__((overloadable, always_inline))
inline uchar8 clz (uchar8 x)
{
  return __builtin_vmp_clz_8(x);
}

__attribute__((overloadable, always_inline))
inline short4 clz (short4 x)
{
  return as_short4(__builtin_vmp_clz_16(as_ushort4(x)));
}

__attribute__((overloadable, always_inline))
inline ushort4 clz (ushort4 x)
{
  return __builtin_vmp_clz_16(x);
}

__attribute__((overloadable, always_inline))
inline int2 clz (int2 x)
{
  return as_int2(__builtin_vmp_clz_32(as_uint2(x)));
}

__attribute__((overloadable, always_inline))
inline uint2 clz (uint2 x)
{
  return __builtin_vmp_clz_32(x);
}

// gentype ctz (gentype x)
__attribute__((overloadable, always_inline))
inline char8 ctz (char8 x)
{
  return as_char8(__builtin_vmp_ctz_8(as_uchar8(x)));
}

__attribute__((overloadable, always_inline))
inline uchar8 ctz (uchar8 x)
{
  return __builtin_vmp_ctz_8(x);
}

__attribute__((overloadable, always_inline))
inline short4 ctz (short4 x)
{
  return as_short4(__builtin_vmp_ctz_16(as_ushort4(x)));
}

__attribute__((overloadable, always_inline))
inline ushort4 ctz (ushort4 x)
{
  return __builtin_vmp_ctz_16(x);
}

__attribute__((overloadable, always_inline))
inline int2 ctz (int2 x)
{
  return as_int2(__builtin_vmp_ctz_32(as_uint2(x)));
}

__attribute__((overloadable, always_inline))
inline uint2 ctz (uint2 x)
{
  return __builtin_vmp_ctz_32(x);
}

// gentype mul_hi (gentype x, gentype y)
__attribute__((overloadable, always_inline))
inline char8 mul_hi (char8 x, char8 y)
{
  short8 mulres = __builtin_vmp_vmacpl0_8(x, y);
  return builtin_shuffle8(as_char8(mulres.lo), as_char8(mulres.hi),
                          1, 3, 5, 7, 9, 11, 13, 15);
}


__attribute__((overloadable, always_inline))
inline uchar8 mul_hi (uchar8 x, uchar8 y)
{
  ushort8 mulres = __builtin_vmp_vmacpl0_u8(x, y);
  return builtin_shuffle8(as_uchar8(mulres.lo), as_uchar8(mulres.hi),
                          1, 3, 5, 7, 9, 11, 13, 15);
}


__attribute__((overloadable, always_inline))
inline short4 mul_hi (short4 x, short4 y)
{
  int4 mulres = __builtin_vmp_vmacpl0_16(x, y);
  return builtin_shuffle4(as_short4(mulres.lo), as_short4(mulres.hi),
                          1, 3, 5, 7);
}


__attribute__((overloadable, always_inline))
inline ushort4 mul_hi (ushort4 x, ushort4 y)
{
  uint4 mulres = __builtin_vmp_vmacpl0_u16(x, y);
  return builtin_shuffle4(as_ushort4(mulres.lo), as_ushort4(mulres.hi),
                          1, 3, 5, 7);
}


__attribute__((overloadable, always_inline))
inline int2 mul_hi (int2 x, int2 y)
{
  long2 mulres = __builtin_vmp_vmacpl0_32(x, y);
  return builtin_shuffle2(as_int2((long1)mulres.lo), as_int2((long1)mulres.hi),
                          1, 3);
}


__attribute__((overloadable, always_inline))
inline uint2 mul_hi (uint2 x, uint2 y)
{
  ulong2 mulres = __builtin_vmp_vmacpl0_u32(x, y);
  return builtin_shuffle2(as_uint2((ulong1)mulres.lo),
                          as_uint2((ulong1)mulres.hi), 1, 3);
}


// gentype mad_hi (gentype a, gentype b, gentype c)
__attribute__((overloadable, always_inline))
inline char8 mad_hi (char8 a, char8 b, char8 c)
{
  return mul_hi(a, b) + c;
}

__attribute__((overloadable, always_inline))
inline uchar8 mad_hi (uchar8 a, uchar8 b, uchar8 c)
{
  return mul_hi(a, b) + c;
}

__attribute__((overloadable, always_inline))
inline short4 mad_hi (short4 a, short4 b, short4 c)
{
  return mul_hi(a, b) + c;
}

__attribute__((overloadable, always_inline))
inline ushort4 mad_hi (ushort4 a, ushort4 b, ushort4 c)
{
  return mul_hi(a, b) + c;
}

__attribute__((overloadable, always_inline))
inline int2 mad_hi (int2 a, int2 b, int2 c)
{
  return mul_hi(a, b) + c;
}

__attribute__((overloadable, always_inline))
inline uint2 mad_hi (uint2 a, uint2 b, uint2 c)
{
  return mul_hi(a, b) + c;
}


/*
 * ****************************************
 * multiply and accumulate
 * ****************************************
 *
 */

// gentype mad_sat (gentype a, gentype b, gentype c)
__attribute__((overloadable, always_inline))
inline short8 mad_sat (char8 a, char8 b, short8 c)
{
  return __builtin_vmp_vmac_8s(c, a, b);
}

__attribute__((overloadable, always_inline))
inline ushort8 mad_sat (uchar8 a, uchar8 b, ushort8 c)
{
  return __builtin_vmp_vmac_u8s(c, a, b);
}

__attribute__((overloadable, always_inline))
inline int4 mad_sat (short4 a, short4 b, int4 c)
{
  return __builtin_vmp_vmac_16s(c, a, b);
}

__attribute__((overloadable, always_inline))
inline uint4 mad_sat (ushort4 a, ushort4 b, uint4 c)
{
  return __builtin_vmp_vmac_u16s(c, a, b);
}

// gentype mad (gentype a, gentype b, gentype c): a * b + c
__attribute__((overloadable, always_inline))
inline short8 mad (char8 a, char8 b, short8 c)
{
  return __builtin_vmp_vmac_8(c, a, b);
}

__attribute__((overloadable, always_inline))
inline ushort8 mad (uchar8 a, uchar8 b, ushort8 c)
{
  return __builtin_vmp_vmac_u8(c, a, b);
}

__attribute__((overloadable, always_inline))
inline int4 mad (short4 a, short4 b, int4 c) {
  return __builtin_vmp_vmac_16(c, a, b);
}

__attribute__((overloadable, always_inline))
inline uint4 mad (ushort4 a, ushort4 b, uint4 c)
{
  return __builtin_vmp_vmac_u16(c, a, b);
}

__attribute__((overloadable, always_inline))
inline long2 mad (int2 a, int2 b, long2 c)
{
  return __builtin_vmp_vmac_32(c, a, b);
}

__attribute__((overloadable, always_inline))
inline ulong2 mad (uint2 a, uint2 b, ulong2 c)
{
  return __builtin_vmp_vmac_u32(c, a, b);
}


// gentype max (gentype x, gentype y) gentype max (gentype x, sgentype y)
__attribute__((overloadable, always_inline))
inline char8 max (char8 x, char8 y)
{
  return __builtin_vmp_vmax_8(x, y);
}

__attribute__((overloadable, always_inline))
inline uchar8 max (uchar8 x, uchar8 y)
{
  return __builtin_vmp_vmax_u8(x, y);
}

__attribute__((overloadable, always_inline))
inline short4 max (short4 x, short4 y)
{
  return __builtin_vmp_vmax_16(x, y);
}

__attribute__((overloadable, always_inline))
inline ushort4 max (ushort4 x, ushort4 y)
{
  return __builtin_vmp_vmax_u16(x, y);
}

__attribute__((overloadable, always_inline))
inline int2 max (int2 x, int2 y)
{
  return __builtin_vmp_vmax_32(x, y);
}

__attribute__((overloadable, always_inline))
inline uint2 max (uint2 x, uint2 y)
{
  return __builtin_vmp_vmax_u32(x, y);
}

// gentype min (gentype x, gentype y) gentype min (gentype x, sgentype y)
__attribute__((overloadable, always_inline))
inline char8 min (char8 x, char8 y)
{
  return __builtin_vmp_vmin_8(x, y);
}

__attribute__((overloadable, always_inline))
inline uchar8 min (uchar8 x, uchar8 y)
{
  return __builtin_vmp_vmin_u8(x, y);
}

__attribute__((overloadable, always_inline))
inline short4 min (short4 x, short4 y)
{
  return __builtin_vmp_vmin_16(x, y);
}

__attribute__((overloadable, always_inline))
inline ushort4 min (ushort4 x, ushort4 y)
{
  return __builtin_vmp_vmin_u16(x, y);
}

__attribute__((overloadable, always_inline))
inline int2 min (int2 x, int2 y)
{
  return __builtin_vmp_vmin_32(x, y);
}

__attribute__((overloadable, always_inline))
inline uint2 min (uint2 x, uint2 y)
{
  return __builtin_vmp_vmin_u32(x, y);
}

// gentype rotate (gentype v, gentype i) (rotate left)
__attribute__((overloadable, always_inline))
inline char8 rotate (char8 x, char8 i)
{
  return (x << i) | convert_char8(convert_uchar8(x) >> convert_uchar8((set_char8(8,8,8,8,8,8,8,8) - i)));
}

__attribute__((overloadable, always_inline))
inline uchar8 rotate (uchar8 x, uchar8 i)
{
  return (x << i) | x >> (set_uchar8(8,8,8,8,8,8,8,8) - i);
}

__attribute__((overloadable, always_inline))
inline short4 rotate (short4 x, short4 i)
{
  return (x << i) | convert_short4(convert_ushort4(x) >> convert_ushort4((set_short4(16,16,16,16) - i)));
}

__attribute__((overloadable, always_inline))
inline ushort4 rotate (ushort4 x, ushort4 i)
{
  return (x << i) | x >> (set_ushort4(16,16,16,16) - i);
}

__attribute__((overloadable, always_inline))
inline int2 rotate (int2 x, int2 i)
{
  return (x << i) | convert_int2(convert_uint2(x) >> convert_uint2((set_int2(32,32) - i)));
}

__attribute__((overloadable, always_inline))
inline uint2 rotate (uint2 x, uint2 i)
{
  return (x << i) | x >> (set_uint2(32,32) - i);
}

// gentype sub_sat (gentype x, gentype y)
__attribute__((overloadable, always_inline))
inline char8 sub_sat (char8 x, char8 y)
{
  return __builtin_vmp_vsub_8s(x, y);
}

__attribute__((overloadable, always_inline))
inline uchar8 sub_sat (uchar8 x, uchar8 y)
{
  return __builtin_vmp_vsub_u8s(x, y);
}

__attribute__((overloadable, always_inline))
inline short4 sub_sat (short4 x, short4 y)
{
  return __builtin_vmp_vsub_16s(x, y);
}

__attribute__((overloadable, always_inline))
inline ushort4 sub_sat (ushort4 x, ushort4 y)
{
  return __builtin_vmp_vsub_u16s(x, y);
}

__attribute__((overloadable, always_inline))
inline int2 sub_sat (int2 x, int2 y)
{
  return __builtin_vmp_vsub_32s(x, y);
}

__attribute__((overloadable, always_inline))
inline uint2 sub_sat (uint2 x, uint2 y)
{
  return __builtin_vmp_vsub_u32s(x, y);
}

// short upsample (char hi, uchar lo)
// ushort upsample (uchar hi, uchar lo)
// shortn upsample (charn hi, ucharn lo)
__attribute__((overloadable, always_inline))
inline short8 upsample (char8 hi, uchar8 lo)
{
  short8 res;
  res.lo =
      as_short4(builtin_shuffle8(lo, as_uchar8(hi), 0, 8, 1, 9, 2, 10, 3, 11));
  res.hi = as_short4(
      builtin_shuffle8(lo, as_uchar8(hi), 4, 12, 5, 13, 6, 14, 7, 15));
  return res;
}

// ushortn upsample (ucharn hi, ucharn lo)
__attribute__((overloadable, always_inline))
inline ushort8 upsample(uchar8 hi, uchar8 lo)
{
  ushort8 res;
  res.lo = as_ushort4(builtin_shuffle8(lo, hi, 0, 8, 1, 9, 2, 10, 3, 11));
  res.hi = as_ushort4(builtin_shuffle8(lo, hi, 4, 12, 5, 13, 6, 14, 7, 15));
  return res;
}

// int upsample (short hi, ushort lo)
// uint upsample (ushort hi, ushort lo)
// intn upsample (shortn hi, ushortn lo)
__attribute__((overloadable, always_inline))
inline int4 upsample (short4 hi, ushort4 lo)
{
  int4 res;
  res.lo = as_int2(builtin_shuffle4(lo, as_ushort4(hi), 0, 4, 1, 5));
  res.hi = as_int2(builtin_shuffle4(lo, as_ushort4(hi), 2, 6, 3, 7));
  return res;
}

// uintn upsample (ushortn hi, ushortn lo)
__attribute__((overloadable, always_inline))
inline uint4 upsample (ushort4 hi, ushort4 lo)
{
  uint4 res;
  res.lo = as_uint2(builtin_shuffle4(lo, hi, 0, 4, 1, 5));
  res.hi = as_uint2(builtin_shuffle4(lo, hi, 2, 6, 3, 7));
  return res;
}

// long upsample (int hi, uint lo)
// ulong upsample (uint hi, uint lo)
// longn upsample (intn hi, uintn lo)
__attribute__((overloadable, always_inline))
inline long2 upsample (int2 hi, uint2 lo)
{
  long2 res;
  res.lo = as_long1(builtin_shuffle2(lo, as_uint2(hi), 0, 2)).s0;
  res.hi = as_long1(builtin_shuffle2(lo, as_uint2(hi), 1, 3)).s0;
  return res;
}

// ulongn upsample (uintn hi, uintn lo)
__attribute__((overloadable, always_inline))
inline ulong2 upsample (uint2 hi, uint2 lo)
{
  ulong2 res;
  res.lo = as_ulong1(builtin_shuffle2(lo, hi, 0, 2)).s0;
  res.hi = as_ulong1(builtin_shuffle2(lo, hi, 1, 3)).s0;
  return res;
}

// gentype popcount (gentype x)
__attribute__((overloadable, always_inline))
inline char8 popcount (char8 x)
{
  return as_char8(__builtin_vmp_popcount_8(as_uchar8(x)));
}

__attribute__((overloadable, always_inline))
inline uchar8 popcount (uchar8 x)
{
  return __builtin_vmp_popcount_8(x);
}

__attribute__((overloadable, always_inline))
inline short4 popcount (short4 x)
{
  return as_short4(__builtin_vmp_popcount_16(as_ushort4(x)));
}

__attribute__((overloadable, always_inline))
inline ushort4 popcount (ushort4 x)
{
  return __builtin_vmp_popcount_16(x);
}

__attribute__((overloadable, always_inline))
inline int2 popcount (int2 x)
{
  return as_int2(__builtin_vmp_popcount_32(as_uint2(x)));
}

__attribute__((overloadable, always_inline))
inline uint2 popcount (uint2 x)
{
  return __builtin_vmp_popcount_32(x);
}

/*
 * ****************************************
 * 6.13.6 relational built-in functions
 * ****************************************
 */

// create relational functions is...
// isequal
__attribute__((overloadable, always_inline))
inline char8 isequal (char8 v0, char8 v1)
{
  return (v0 == v1);
}

__attribute__((overloadable, always_inline))
inline short4 isequal (short4 v0, short4 v1)
{
  return (v0 == v1);
}

__attribute__((overloadable, always_inline))
inline int2 isequal (int2 v0, int2 v1)
{
  return (v0 == v1);
}

__attribute__((overloadable, always_inline))
inline char8 isequal (uchar8 v0, uchar8 v1)
{
  return (v0 == v1);
}

__attribute__((overloadable, always_inline))
inline short4 isequal (ushort4 v0, ushort4 v1)
{
  return (v0 == v1);
}

__attribute__((overloadable, always_inline))
inline int2 isequal (uint2 v0, uint2 v1)
{
  return (v0 == v1);
}

// isnotequal
__attribute__((overloadable, always_inline))
inline char8 isnotequal (char8 v0, char8 v1)
{
  return (v0 != v1);
}

__attribute__((overloadable, always_inline))
inline short4 isnotequal (short4 v0, short4 v1)
{
  return (v0 != v1);
}

__attribute__((overloadable, always_inline))
inline int2 isnotequal (int2 v0, int2 v1)
{
  return (v0 != v1);
}

__attribute__((overloadable, always_inline))
inline char8 isnotequal (uchar8 v0, uchar8 v1)
{
  return (v0 != v1);
}

__attribute__((overloadable, always_inline))
inline short4 isnotequal (ushort4 v0, ushort4 v1)
{
  return (v0 != v1);
}

__attribute__((overloadable, always_inline))
inline int2 isnotequal (uint2 v0, uint2 v1)
{
  return (v0 != v1);
}

// isgreater
__attribute__((overloadable, always_inline))
inline char8 isgreater (char8 v0, char8 v1)
{
  return (v0 > v1);
}

__attribute__((overloadable, always_inline))
inline short4 isgreater (short4 v0, short4 v1)
{
  return (v0 > v1);
}

__attribute__((overloadable, always_inline))
inline int2 isgreater (int2 v0, int2 v1)
{
  return (v0 > v1);
}

__attribute__((overloadable, always_inline))
inline char8 isgreater (uchar8 v0, uchar8 v1)
{
  return (v0 > v1);
}

__attribute__((overloadable, always_inline))
inline short4 isgreater (ushort4 v0, ushort4 v1)
{
  return (v0 > v1);
}

__attribute__((overloadable, always_inline))
inline int2 isgreater (uint2 v0, uint2 v1)
{
  return (v0 > v1);
}

// isgreaterequal
__attribute__((overloadable, always_inline))
inline char8 isgreaterequal (char8 v0, char8 v1)
{
  return (v0 >= v1);
}

__attribute__((overloadable, always_inline))
inline short4 isgreaterequal (short4 v0, short4 v1)
{
  return (v0 >= v1);
}

__attribute__((overloadable, always_inline))
inline int2 isgreaterequal (int2 v0, int2 v1)
{
  return (v0 >= v1);
}

__attribute__((overloadable, always_inline))
inline char8 isgreaterequal (uchar8 v0, uchar8 v1)
{
  return (v0 >= v1);
}

__attribute__((overloadable, always_inline))
inline short4 isgreaterequal (ushort4 v0, ushort4 v1)
{
  return (v0 >= v1);
}

__attribute__((overloadable, always_inline))
inline int2 isgreaterequal (uint2 v0, uint2 v1)
{
  return (v0 >= v1);
}

// isless
__attribute__((overloadable, always_inline))
inline char8 isless (char8 v0, char8 v1)
{
  return (v0 < v1);
}

__attribute__((overloadable, always_inline))
inline short4 isless (short4 v0, short4 v1)
{
  return (v0 < v1);
}

__attribute__((overloadable, always_inline))
inline int2 isless (int2 v0, int2 v1)
{
  return (v0 < v1);
}

__attribute__((overloadable, always_inline))
inline char8 isless (uchar8 v0, uchar8 v1)
{
  return (v0 < v1);
}

__attribute__((overloadable, always_inline))
inline short4 isless (ushort4 v0, ushort4 v1)
{
  return (v0 < v1);
}

__attribute__((overloadable, always_inline))
inline int2 isless (uint2 v0, uint2 v1)
{
  return (v0 < v1);
}

// islessequal
__attribute__((overloadable, always_inline))
inline char8 islessequal (char8 v0, char8 v1)
{
  return (v0 <= v1);
}

__attribute__((overloadable, always_inline))
inline short4 islessequal (short4 v0, short4 v1)
{
  return (v0 <= v1);
}

__attribute__((overloadable, always_inline))
inline int2 islessequal (int2 v0, int2 v1)
{
  return (v0 <= v1);
}

__attribute__((overloadable, always_inline))
inline char8 islessequal (uchar8 v0, uchar8 v1)
{
  return (v0 <= v1);
}

__attribute__((overloadable, always_inline))
inline short4 islessequal (ushort4 v0, ushort4 v1)
{
  return (v0 <= v1);
}

__attribute__((overloadable, always_inline))
inline int2 islessequal (uint2 v0, uint2 v1)
{
  return (v0 <= v1);
}

// signbit
__attribute__((overloadable, always_inline))
inline char8 signbit (char8 a)
{
  return isless(a, set_char8(0,0,0,0,0,0,0,0));
}

__attribute__((overloadable, always_inline))
inline short4 signbit (short4 a)
{
  return isless(a, set_short4(0,0,0,0));
}

__attribute__((overloadable, always_inline))
inline int2 signbit (int2 a)
{
  return isless(a, set_int2(0,0));
}

// any_masked
__attribute__((overloadable, always_inline))
inline int any_masked (char8 v0, uchar8 mask)
{
  return __builtin_vmp_any_8(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int any_masked (short4 v0, ushort4 mask)
{
  return __builtin_vmp_any_16(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int any_masked (int2 v0, uint2 mask)
{
  return __builtin_vmp_any_32(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int any_masked (uchar8 v0, uchar8 mask)
{
  return __builtin_vmp_any_u8(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int any_masked (ushort4 v0, ushort4 mask)
{
  return __builtin_vmp_any_u16(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int any_masked (uint2 v0, uint2 mask)
{
  return __builtin_vmp_any_u32(v0, mask);
}

// any
__attribute__((overloadable, always_inline))
inline int any (char8 v0)
{
  return any_masked(v0, set_uchar8(1u));
}

__attribute__((overloadable, always_inline))
inline int any (short4 v0)
{
  return any_masked(v0, set_ushort4(1u));
}

__attribute__((overloadable, always_inline))
inline int any (int2 v0)
{
  return any_masked(v0, set_uint2(1u));
}

__attribute__((overloadable, always_inline))
inline int any (uchar8 v0)
{
  return any_masked(v0, set_uchar8(1u));
}

__attribute__((overloadable, always_inline))
inline int any (ushort4 v0)
{
  return any_masked(v0, set_ushort4(1u));
}

__attribute__((overloadable, always_inline))
inline int any (uint2 v0)
{
  return any_masked(v0, set_uint2(1u));
}

// all_masked
__attribute__((overloadable, always_inline))
inline int all_masked (char8 v0, uchar8 mask)
{
  return __builtin_vmp_all_8(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int all_masked (short4 v0, ushort4 mask)
{
  return __builtin_vmp_all_16(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int all_masked (int2 v0, uint2 mask)
{
  return __builtin_vmp_all_32(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int all_masked (uchar8 v0, uchar8 mask)
{
  return __builtin_vmp_all_u8(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int all_masked (ushort4 v0, ushort4 mask)
{
  return __builtin_vmp_all_u16(v0, mask);
}

__attribute__((overloadable, always_inline))
inline int all_masked (uint2 v0, uint2 mask)
{
  return __builtin_vmp_all_u32(v0, mask);
}

// all
__attribute__((overloadable, always_inline))
inline int all (char8 v0)
{
  return all_masked(v0, set_uchar8(1u));
}

__attribute__((overloadable, always_inline))
inline int all (short4 v0)
{
  return all_masked(v0, set_ushort4(1u));
}

__attribute__((overloadable, always_inline))
inline int all (int2 v0)
{
  return all_masked(v0, set_uint2(1u));
}

__attribute__((overloadable, always_inline))
inline int all (uchar8 v0)
{
  return all_masked(v0, set_uchar8(1u));
}

__attribute__((overloadable, always_inline))
inline int all (ushort4 v0)
{
  return all_masked(v0, set_ushort4(1u));
}

__attribute__((overloadable, always_inline))
inline int all (uint2 v0)
{
  return all_masked(v0, set_uint2(1u));
}

// bitselect
__attribute__((overloadable, always_inline))
inline char8 bitselect (char8 a, char8 b, char8 c)
{
  return ((b & c) | ( a & ~c));
}

__attribute__((overloadable, always_inline))
inline short4 bitselect (short4 a, short4 b, short4 c)
{
  return ((b & c) | ( a & ~c));
}

__attribute__((overloadable, always_inline))
inline int2 bitselect (int2 a, int2 b, int2 c)
{
  return ((b & c) | ( a & ~c));
}

__attribute__((overloadable, always_inline))
inline uchar8 bitselect (uchar8 a, uchar8 b, uchar8 c)
{
  return ((b & c) | ( a & ~c));
}

__attribute__((overloadable, always_inline))
inline ushort4 bitselect (ushort4 a, ushort4 b, ushort4 c)
{
  return ((b & c) | ( a & ~c));
}

__attribute__((overloadable, always_inline))
inline uint2 bitselect (uint2 a, uint2 b, uint2 c)
{
  return ((b & c) | ( a & ~c));
}

// select
__attribute__((overloadable, always_inline))
inline char8 select (char8 f, char8 t, uchar8 s)
{
  return __builtin_vmp_vselect_8(f, t, s);
}

__attribute__((overloadable, always_inline))
inline uchar8 select (uchar8 f, uchar8 t, uchar8 s)
{
  return convert_uchar8(__builtin_vmp_vselect_8(convert_char8(f), convert_char8(t), s));
}

__attribute__((overloadable, always_inline))
inline short4 select (short4 f, short4 t, ushort4 s)
{
  return __builtin_vmp_vselect_16(f, t, s);
}

__attribute__((overloadable, always_inline))
inline ushort4 select (ushort4 f, ushort4 t, ushort4 s)
{
  return convert_ushort4(__builtin_vmp_vselect_16(convert_short4(f), convert_short4(t), s));
}

__attribute__((overloadable, always_inline))
inline int2 select (int2 f, int2 t, uint2 s)
{
  return __builtin_vmp_vselect_32(f, t, s);
}

__attribute__((overloadable, always_inline))
inline uint2 select (uint2 f, uint2 t, uint2 s)
{
  return convert_uint2(__builtin_vmp_vselect_32(convert_int2(f), convert_int2(t), s));
}

__attribute__((overloadable, always_inline))
inline char8 select (char8 f, char8 t, char8 s)
{
  return __builtin_vmp_vselect_8(f, t, convert_uchar8(s));
}

__attribute__((overloadable, always_inline))
inline uchar8 select (uchar8 f, uchar8 t, char8 s)
{
  return convert_uchar8(__builtin_vmp_vselect_8(convert_char8(f),
                                                convert_char8(t),
                                                convert_uchar8(s)));
}

__attribute__((overloadable, always_inline))
inline short4 select (short4 f, short4 t, short4 s)
{
  return __builtin_vmp_vselect_16(f, t, convert_ushort4(s));
}

__attribute__((overloadable, always_inline))
inline ushort4 select (ushort4 f, ushort4 t, short4 s)
{
  return convert_ushort4(__builtin_vmp_vselect_16(convert_short4(f),
                                                  convert_short4(t),
                                                  convert_ushort4(s)));
}

__attribute__((overloadable, always_inline))
inline int2 select (int2 f, int2 t, int2 s)
{
  return __builtin_vmp_vselect_32(f, t, convert_uint2(s));
}

__attribute__((overloadable, always_inline))
inline uint2 select (uint2 f, uint2 t, int2 s)
{
  return convert_uint2(__builtin_vmp_vselect_32(convert_int2(f),
                                                convert_int2(t),
                                                convert_uint2(s)));
}


/*
 * ****************************************
 * permuting a single vector
 * ****************************************
 */

// permute v8i8
__attribute__((overloadable, always_inline))
inline char8 shuffle (char8 vector1, uchar8 mask)
{
  return __builtin_vmp_vperm_8(vector1, mask);
}

__attribute__((overloadable, always_inline))
inline uchar8 shuffle (uchar8 vector1, uchar8 mask)
{
  return convert_uchar8(__builtin_vmp_vperm_8(convert_char8(vector1), mask));
}


// permute v4i16
__attribute__((overloadable, always_inline))
inline short4 shuffle (short4 vector1, ushort4 mask)
{
  return __builtin_vmp_vperm_16(vector1, mask);
}

__attribute__((overloadable, always_inline))
inline ushort4 shuffle (ushort4 vector1, ushort4 mask)
{
  return convert_ushort4(__builtin_vmp_vperm_16(convert_short4(vector1), mask));
}


// permute v2i32
__attribute__((overloadable, always_inline))
inline int2 shuffle (int2 vector1, uint2 mask)
{
  return __builtin_vmp_vperm_32(vector1, mask);
}

__attribute__((overloadable, always_inline))
inline uint2 shuffle (uint2 vector1, uint2 mask)
{
  return convert_uint2(__builtin_vmp_vperm_32(convert_int2(vector1), mask));
}


/*
 * ****************************************
 * shuffling two vectors
 * ****************************************
 */

// shuffle v8i8
__attribute__((overloadable, always_inline))
inline char8 shuffle2 (char8 vector1, char8 vector2, uchar8 mask)
{
  return __builtin_vmp_vpermreg_8(vector1, vector2, mask);
}

__attribute__((overloadable, always_inline))
inline uchar8 shuffle2 (uchar8 vector1, uchar8 vector2, uchar8 mask)
{
  return as_uchar8(__builtin_vmp_vpermreg_8(as_char8(vector1),
                                            as_char8(vector2),
                                            mask));
}


// shuffle v4i16
__attribute__((overloadable, always_inline))
inline short4 shuffle2 (short4 vector1, short4 vector2, ushort4 mask)
{
  return __builtin_vmp_vpermreg_16(vector1, vector2, mask);
}

__attribute__((overloadable, always_inline))
inline ushort4 shuffle2 (ushort4 vector1, ushort4 vector2, ushort4 mask)
{
  return as_ushort4(__builtin_vmp_vpermreg_16(as_short4(vector1),
                                              as_short4(vector2),
                                              mask));
}


// shuffle v2i32
__attribute__((overloadable, always_inline))
inline int2 shuffle2 (int2 vector1, int2 vector2, uint2 mask)
{
  return __builtin_vmp_vpermreg_32(vector1, vector2, mask);
}

__attribute__((overloadable, always_inline))
inline uint2 shuffle2 (uint2 vector1, uint2 vector2, uint2 mask)
{
  return as_uint2(__builtin_vmp_vpermreg_32(as_int2(vector1),
                                            as_int2(vector2),
                                            mask));
}


# endif // __VMP_CL_FUNCTIONS_H__

