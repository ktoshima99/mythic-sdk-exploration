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
* FILENAME: vid_videantis-C.hpp
*
* DESCRIPTION: videantis-C language mapping to C++
*              used to make data types and keywords available
*              to static code analysis with FlexeLint and oclint
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef VID_VIDEANTISC_HPP_
#define VID_VIDEANTISC_HPP_

//lint -save
//lint -e9* -e621 -e553 -e808 -e40 -e10 -e19
//lintReason: These MISRA rules are invalid for gcc standard include files
//lintReason: gcc standard include files required for PC-Lint
#ifdef _PCLINT
#include <cstddef>
#endif
//lint -restore

//lint -save

//lint -e970
//lintNote 970: Use of modifier or type 'int' outside of a typedef [MISRA 2012 Directive 4.6, advisory]
//lintReason: OpenCL-C builtin types have defined fixed bit widths

//lint -e9026 Note 9026: Function-like macro defined [MISRA 2012 Directive 4.9, advisory]
//lintReason Lint exceptions introduced to enable C++ template like functions enabled by macros

// CL language
#define __LOCAL __local
#define __KERNEL __kernel

#define  __builtin_vmp_convert_wordaddresstobyteaddress
#define  __builtin_vmp_convert_byteaddresstowordaddress

#define __byteaddress
#define __wordaddress

#define __WORDADDRESS
#define __BYTEADDRESS

#define __constant const


#define SIZEOF_IN_WORDS(_type) (sizeof(_type) / 8u)
#define SIZEOF_IN_BYTES(_type) sizeof(_type)


/*
 * scalar types
 * for compatibility reasons most are defined, even if they are not supported
 */
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;



#define CHAR_BIT         8
#define SCHAR_MAX        127
#define SCHAR_MIN        (-127-1)
#define CHAR_MAX         SCHAR_MAX
#define CHAR_MIN         SCHAR_MIN
#define UCHAR_MIN        0
#define UCHAR_MAX        255
#define SHRT_MAX         32767
#define SHRT_MIN         (-32767-1)
#define USHRT_MIN        0
#define USHRT_MAX        65535
#define INT_MAX          2147483647
#define INT_MIN          (-2147483647-1)
#define UINT_MIN         0
#define UINT_MAX         0xffffffffU
#define LONG_MAX         ((long) 0x7FFFFFFFFFFFFFFFLL)
#define LONG_MIN         ((long) -0x7FFFFFFFFFFFFFFFLL - 1LL)
#define ULONG_MIN        0
#define ULONG_MAX        ((ulong) 0xFFFFFFFFFFFFFFFFULL)

#define __global
#define __local
#define __constant const
#define __private
#define __kernel

#define global
#define local
#define constant const
//#define private
#define kernel


/* v-MP native vector data types */

#define VMP_VECTOR_CLASS(T, Th, Ts) \
\
class T \
{ \
public: \
  T(void); \
  T(const Ts a); \
  T(const T &a); \
  T &operator%=(const T &a); \
  T &operator&=(const T &a); \
  T &operator*=(const T &a); \
  T &operator+=(const T &a); \
  T &operator-=(const T &a); \
  T &operator/=(const T &a); \
  T &operator^=(const T &a); \
  T &operator|=(const T &a); \
  Ts x, y, w, z; \
  Th hi, lo; \
  Th even, odd; \
  Th s00, s01, s10, s11; \
  Ts s0000, s1111, s2222, s3333; \
  Ts xx, xy, yx, yy; \
  Ts s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sa, sb, sc, sd, se, sf; \
};



VMP_VECTOR_CLASS(long1, long, long)
VMP_VECTOR_CLASS(int2, int, int)
VMP_VECTOR_CLASS(short2, short, short)
VMP_VECTOR_CLASS(short4, short2, short)
VMP_VECTOR_CLASS(char2, char, char)
VMP_VECTOR_CLASS(char4, char2, char)
VMP_VECTOR_CLASS(char8, char4, char)
VMP_VECTOR_CLASS(ulong1, ulong, ulong)
VMP_VECTOR_CLASS(uint2, uint, uint)
VMP_VECTOR_CLASS(ushort2, ushort, ushort)
VMP_VECTOR_CLASS(ushort4, ushort2, ushort)
VMP_VECTOR_CLASS(uchar2, uchar, uchar)
VMP_VECTOR_CLASS(uchar4, uchar2, uchar)
VMP_VECTOR_CLASS(uchar8, uchar4, uchar)
VMP_VECTOR_CLASS(long2, long, long)
VMP_VECTOR_CLASS(int4, int2, int)
VMP_VECTOR_CLASS(short8, short4, short)
VMP_VECTOR_CLASS(char16, char8, char)
VMP_VECTOR_CLASS(ulong2, ulong, ulong)
VMP_VECTOR_CLASS(uint4, uint2, uint)
VMP_VECTOR_CLASS(ushort8, ushort4, ushort)
VMP_VECTOR_CLASS(uchar16, uchar8, uchar)

// set functions broadcast to native vector with single argument
//FIXME: MISRA fixes in v-MathLib required
long1 set_long1(int s0);
long1 set_long1(long s0);

int2 set_int2(int s0);
short4 set_short4(short s0);
char8 set_char8(char s0);

//FIXME: MISRA fixes in v-MathLib required
ulong1 set_ulong1(int s0);
ulong1 set_ulong1(uint s0);
uint2 set_uint2(int s0);


ulong1 set_ulong1(ulong s0);
uint2 set_uint2(uint s0);
ushort4 set_ushort4(ushort s0);
uchar8 set_uchar8(uchar s0);


// set functions with argument for each native vector element
int2 set_int2(int s0, int s1);
short4 set_short4(short2 s0, short2 s1);
short4 set_short4(short s0, short s1, short s2, short s3);
char8 set_char8(char s0, char s1, char s2, char s3, char s4, char s5, char s6, char s7);

uint2 set_uint2(uint s0, uint s1);
ushort4 set_ushort4(ushort2 s0, ushort2 s1);
ushort4 set_ushort4(ushort s0, ushort s1, ushort s2, ushort s3);
uchar8 set_uchar8(uchar4 s0, uchar4 s1);
uchar8 set_uchar8(uchar s0, uchar s1, uchar s2, uchar s3, uchar s4, uchar s5, uchar s6, uchar s7);


// set functions broadcast to double vector with single argument
long2 set_long2(long s0);
int4 set_int4(int s0);
short8 set_short8(short s0);
char16 set_char16(char s0);

ulong2 set_ulong2(ulong s0);
uint4 set_uint4(uint s0);
ushort8 set_ushort8(ushort s0);
uchar16 set_uchar16(uchar s0);


// set functions with argument for each double vector element
long2 set_long2(long s0, long s1);
int4 set_int4(int s0, int s1, int s2, int s3);
short8 set_short8(short s0, short s1, short s2, short s3, short s4, short s5, short s6, short s7);
char16 set_char16(char s0, char s1, char s2, char s3, char s4, char s5, char s6, char s7, char s8, char s9, char s10, char s11, char s12, char s13, char s14, char s15);

ulong2 set_ulong2(ulong s0, ulong s1);
uint4 set_uint4(uint s0, uint s1, uint s2, uint s3);
ushort8 set_ushort8(ushort s0, ushort s1, ushort s2, ushort s3, ushort s4, ushort s5, ushort s6, ushort s7);
uchar16 set_uchar16(uchar s0, uchar s1, uchar s2, uchar s3, uchar s4, uchar s5, uchar s6, uchar s7, uchar s8, uchar s9, uchar s10, uchar s11, uchar s12, uchar s13, uchar s14, uchar s15);


ushort8 convert_ushort8(const short8& x);
uint4 convert_uint4(const int4& x);
ulong2 convert_ulong2(const long2& x);


uchar8 convert_uchar8(const char8& x);
ushort4 convert_ushort4(const short4& x);
uint2 convert_uint2(const int2& x);
ulong1 convert_ulong1(const long1& x);
char8 convert_char8(const uchar8& x);
short4 convert_short4(const ushort4& x);
int2 convert_int2(const uint2& x);
long1 convert_long1(const ulong1& x);
short8 convert_short8(const char8& x);
int4 convert_int4(const short4& x);
long2 convert_long2(const int2& x);
ushort8 convert_ushort8(const uchar8& x);
uint4 convert_uint4(const ushort4& x);
ulong2 convert_ulong2(const uint2& x);
char8 convert_char8(const short8& x);
short4 convert_short4(const int4& x);
int2 convert_int2(const long2& x);
uchar8 convert_uchar8(const ushort8& x);
ushort4 convert_ushort4(const uint4& x);
uint2 convert_uint2(const ulong2& x);


long1 operator+(const long1& a, const long1& b);
int2 operator+(const int2& a, const int2& b);
short4 operator+(const short4& a, const short4& b);
char8 operator+(const char8& a, const char8& b);
ulong1 operator+(const ulong1& a, const ulong1& b);
uint2 operator+(const uint2& a, const uint2& b);
ushort4 operator+(const ushort4& a, const ushort4& b);
uchar8 operator+(const uchar8& a, const uchar8& b);

long1 operator-(const long1& a, const long1& b);
int2 operator-(const int2& a, const int2& b);
short4 operator-(const short4& a, const short4& b);
char8 operator-(const char8& a, const char8& b);
ulong1 operator-(const ulong1& a, const ulong1& b);
uint2 operator-(const uint2& a, const uint2& b);
ushort4 operator-(const ushort4& a, const ushort4& b);
uchar8 operator-(const uchar8& a, const uchar8& b);

long1 operator*(const long1& a, const long1& b);
int2 operator*(const int2& a, const int2& b);
short4 operator*(const short4& a, const short4& b);
char8 operator*(const char8& a, const char8& b);
ulong1 operator*(const ulong1& a, const ulong1& b);
uint2 operator*(const uint2& a, const uint2& b);
ushort4 operator*(const ushort4& a, const ushort4& b);
uchar8 operator*(const uchar8& a, const uchar8& b);

long1 operator/(const long1& a, const long1& b);
int2 operator/(const int2& a, const int2& b);
short4 operator/(const short4& a, const short4& b);
char8 operator/(const char8& a, const char8& b);
ulong1 operator/(const ulong1& a, const ulong1& b);
uint2 operator/(const uint2& a, const uint2& b);
ushort4 operator/(const ushort4& a, const ushort4& b);
uchar8 operator/(const uchar8& a, const uchar8& b);

long1 operator%(const long1& a, const long1& b);
int2 operator%(const int2& a, const int2& b);
short4 operator%(const short4& a, const short4& b);
char8 operator%(const char8& a, const char8& b);
ulong1 operator%(const ulong1& a, const ulong1& b);
uint2 operator%(const uint2& a, const uint2& b);
ushort4 operator%(const ushort4& a, const ushort4& b);
uchar8 operator%(const uchar8& a, const uchar8& b);

long1 operator&(const long1& a, const long1& b);
int2 operator&(const int2& a, const int2& b);
short4 operator&(const short4& a, const short4& b);
char8 operator&(const char8& a, const char8& b);
ulong1 operator&(const ulong1& a, const ulong1& b);
uint2 operator&(const uint2& a, const uint2& b);
ushort4 operator&(const ushort4& a, const ushort4& b);
uchar8 operator&(const uchar8& a, const uchar8& b);

long1 operator|(const long1& a, const long1& b);
int2 operator|(const int2& a, const int2& b);
short4 operator|(const short4& a, const short4& b);
char8 operator|(const char8& a, const char8& b);
ulong1 operator|(const ulong1& a, const ulong1& b);
uint2 operator|(const uint2& a, const uint2& b);
ushort4 operator|(const ushort4& a, const ushort4& b);
uchar8 operator|(const uchar8& a, const uchar8& b);

long1 operator^(const long1& a, const long1& b);
int2 operator^(const int2& a, const int2& b);
short4 operator^(const short4& a, const short4& b);
char8 operator^(const char8& a, const char8& b);
ulong1 operator^(const ulong1& a, const ulong1& b);
uint2 operator^(const uint2& a, const uint2& b);
ushort4 operator^(const ushort4& a, const ushort4& b);
uchar8 operator^(const uchar8& a, const uchar8& b);

long1 operator>>(const long1& a, const long1& b);
int2 operator>>(const int2& a, const int2& b);
short4 operator>>(const short4& a, const short4& b);
char8 operator>>(const char8& a, const char8& b);
ulong1 operator>>(const ulong1& a, const ulong1& b);
uint2 operator>>(const uint2& a, const uint2& b);
ushort4 operator>>(const ushort4& a, const ushort4& b);
uchar8 operator>>(const uchar8& a, const uchar8& b);

long1 operator<<(const long1& a, const long1& b);
int2 operator<<(const int2& a, const int2& b);
short4 operator<<(const short4& a, const short4& b);
char8 operator<<(const char8& a, const char8& b);
ulong1 operator<<(const ulong1& a, const ulong1& b);
uint2 operator<<(const uint2& a, const uint2& b);
ushort4 operator<<(const ushort4& a, const ushort4& b);
uchar8 operator<<(const uchar8& a, const uchar8& b);

long1 operator==(const long1& a, const long1& b);
int2 operator==(const int2& a, const int2& b);
short4 operator==(const short4& a, const short4& b);
char8 operator==(const char8& a, const char8& b);
long1 operator==(const ulong1& a, const ulong1& b);
int2 operator==(const uint2& a, const uint2& b);
short4 operator==(const ushort4& a, const ushort4& b);
char8 operator==(const uchar8& a, const uchar8& b);

long1 operator!=(const long1& a, const long1& b);
int2 operator!=(const int2& a, const int2& b);
short4 operator!=(const short4& a, const short4& b);
char8 operator!=(const char8& a, const char8& b);
long1 operator!=(const ulong1& a, const ulong1& b);
int2 operator!=(const uint2& a, const uint2& b);
short4 operator!=(const ushort4& a, const ushort4& b);
char8 operator!=(const uchar8& a, const uchar8& b);

long1 operator&&(const long1& a, const long1& b);
int2 operator&&(const int2& a, const int2& b);
short4 operator&&(const short4& a, const short4& b);
char8 operator&&(const char8& a, const char8& b);
long1 operator&&(const ulong1& a, const ulong1& b);
int2 operator&&(const uint2& a, const uint2& b);
short4 operator&&(const ushort4& a, const ushort4& b);
char8 operator&&(const uchar8& a, const uchar8& b);

long1 operator||(const long1& a, const long1& b);
int2 operator||(const int2& a, const int2& b);
short4 operator||(const short4& a, const short4& b);
char8 operator||(const char8& a, const char8& b);
long1 operator||(const ulong1& a, const ulong1& b);
int2 operator||(const uint2& a, const uint2& b);
short4 operator||(const ushort4& a, const ushort4& b);
char8 operator||(const uchar8& a, const uchar8& b);

long1 operator>(const long1& a, const long1& b);
int2 operator>(const int2& a, const int2& b);
short4 operator>(const short4& a, const short4& b);
char8 operator>(const char8& a, const char8& b);
long1 operator>(const ulong1& a, const ulong1& b);
int2 operator>(const uint2& a, const uint2& b);
short4 operator>(const ushort4& a, const ushort4& b);
char8 operator>(const uchar8& a, const uchar8& b);

long1 operator>=(const long1& a, const long1& b);
int2 operator>=(const int2& a, const int2& b);
short4 operator>=(const short4& a, const short4& b);
char8 operator>=(const char8& a, const char8& b);
long1 operator>=(const ulong1& a, const ulong1& b);
int2 operator>=(const uint2& a, const uint2& b);
short4 operator>=(const ushort4& a, const ushort4& b);
char8 operator>=(const uchar8& a, const uchar8& b);

long1 operator<(const long1& a, const long1& b);
int2 operator<(const int2& a, const int2& b);
short4 operator<(const short4& a, const short4& b);
char8 operator<(const char8& a, const char8& b);
long1 operator<(const ulong1& a, const ulong1& b);
int2 operator<(const uint2& a, const uint2& b);
short4 operator<(const ushort4& a, const ushort4& b);
char8 operator<(const uchar8& a, const uchar8& b);

long1 operator<=(const long1& a, const long1& b);
int2 operator<=(const int2& a, const int2& b);
short4 operator<=(const short4& a, const short4& b);
char8 operator<=(const char8& a, const char8& b);
long1 operator<=(const ulong1& a, const ulong1& b);
int2 operator<=(const uint2& a, const uint2& b);
short4 operator<=(const ushort4& a, const ushort4& b);
char8 operator<=(const uchar8& a, const uchar8& b);

long1& operator!(const long1& x);
int2& operator!(const int2& x);
short4& operator!(const short4& x);
char8& operator!(const char8& x);
long1& operator!(const ulong1& x);
int2& operator!(const uint2& x);
short4& operator!(const ushort4& x);
char8& operator!(const uchar8& x);

long1& operator~(const long1& x);
int2& operator~(const int2& x);
short4& operator~(const short4& x);
char8& operator~(const char8& x);
ulong1& operator~(const ulong1& x);
uint2& operator~(const uint2& x);
ushort4& operator~(const ushort4& x);
uchar8& operator~(const uchar8& x);

long1& operator-(const long1& x);
int2& operator-(const int2& x);
short4& operator-(const short4& x);
char8& operator-(const char8& x);
ulong1& operator-(const ulong1& x);
uint2& operator-(const uint2& x);
ushort4& operator-(const ushort4& x);
uchar8& operator-(const uchar8& x);

//The following builtins write to the STORE_HIGH register or read from the LOAD_HIGH register.
void __builtin_vmp_write_STORE_HIGH(int x);
int __builtin_vmp_read_LOAD_HIGH(void);

//The following builtins perform signed saturated and unsigned saturated ADD instructions on the scalar data path.
int __builtin_vmp_add_s(int x, int y);
uint __builtin_vmp_add_us(uint x, uint y);

//The following builtins perform signed saturated and unsigned saturated SUB instructions on the scalar data path.
int __builtin_vmp_sub_s(int x, int y);
uint __builtin_vmp_sub_us(uint x, uint y);

//The following builtins perform signed and unsigned RND0 and RND00 instructions on the scalar data path.
int __builtin_vmp_rnd0(int x, int y);
uint __builtin_vmp_rnd0_u(uint x, uint y);
int __builtin_vmp_rnd00(int x, int y);
uint __builtin_vmp_rnd00_u(uint x, uint y);

//The following builtins perform signed and unsigned MIN and MAX instructions on the scalar data path.
int __builtin_vmp_min(int x, int y);
uint __builtin_vmp_min_u(uint x, uint y);
int __builtin_vmp_max(int x, int y);
uint __builtin_vmp_max_u(uint x, uint y);

//The following builtin performs a WAIT/SLEEP/BREAK instruction on the scalar data path. The builtin operand has to be an immediate value.
void __builtin_vmp_wait(const uint n);
void __builtin_vmp_sleep(const uint n);
void __builtin_vmp_break(const uint n);

//The following builtins perform a V_ADD instruction on the vector data path in various sizes.
char8 __builtin_vmp_vadd_8s(char8 x, char8 y);
short4 __builtin_vmp_vadd_16s(short4 x, short4 y);
int2 __builtin_vmp_vadd_32s(int2 x, int2 y);
uchar8 __builtin_vmp_vadd_u8s(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vadd_u16s(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vadd_u32s(uint2 x, uint2 y);

//The following builtins perform a V_SUB instruction on the vector data path in various sizes.
char8 __builtin_vmp_vsub_8s(char8 x, char8 y);
short4 __builtin_vmp_vsub_16s(short4 x, short4 y);
int2 __builtin_vmp_vsub_32s(int2 x, int2 y);
uchar8 __builtin_vmp_vsub_u8s(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vsub_u16s(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vsub_u32s(uint2 x, uint2 y);

//The following builtins perform a V_MUL instruction on the vector data path in various sizes.
char8 __builtin_vmp_vmul_8s(char8 x, char8 y);
short4 __builtin_vmp_vmul_16s(short4 x, short4 y);
uchar8 __builtin_vmp_vmul_u8s(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vmul_u16s(ushort4 x, ushort4 y);
//Only saturated builtins are supported, because unsaturated multiplications are already supported by
//the C language.

//The following builtins perform a V_MACPL0 instruction on the vector data path in various sizes.
short8 __builtin_vmp_vmacpl0_8(char8 x, char8 y);
int4 __builtin_vmp_vmacpl0_16(short4 x, short4 y);
long2 __builtin_vmp_vmacpl0_32(int2 x, int2 y);
ushort8 __builtin_vmp_vmacpl0_u8(uchar8 x, uchar8 y);
uint4 __builtin_vmp_vmacpl0_u16(ushort4 x, ushort4 y);
ulong2 __builtin_vmp_vmacpl0_u32(uint2 x, uint2 y);

//The following builtins perform a saturated V_MACPL0 instruction on the vector data path in various sizes.
short8 __builtin_vmp_vmacpl0_8s(char8 x, char8 y);
int4 __builtin_vmp_vmacpl0_16s(short4 x, short4 y);
ushort8 __builtin_vmp_vmacpl0_u8s(uchar8 x, uchar8 y);
uint4 __builtin_vmp_vmacpl0_u16s(ushort4 x, ushort4 y);

//The following builtins perform a V_MAC instruction on the vector data path in various sizes.
short8 __builtin_vmp_vmac_8(short8 accu, char8 x, char8 y);
int4 __builtin_vmp_vmac_16(int4 accu, short4 x, short4 y);
long2 __builtin_vmp_vmac_32(long2 accu, int2 x, int2 y);
ushort8 __builtin_vmp_vmac_u8(ushort8 accu, uchar8 x, uchar8 y);
uint4 __builtin_vmp_vmac_u16(uint4 accu, ushort4 x, ushort4 y);
ulong2 __builtin_vmp_vmac_u32(ulong2 accu, uint2 x, uint2 y);

//The following builtins perform a saturated V_MAC instruction on the vector data path in various sizes.
short8 __builtin_vmp_vmac_8s(short8 accu, char8 x, char8 y);
int4 __builtin_vmp_vmac_16s(int4 accu, short4 x, short4 y);
ushort8 __builtin_vmp_vmac_u8s(ushort8 accu, uchar8 x, uchar8 y);
uint4 __builtin_vmp_vmac_u16s(uint4 accu, ushort4 x, ushort4 y);

//The following builtins perform a V_RND0 instruction on the vector data path in various sizes.
char8 __builtin_vmp_vrnd0_8(char8 x, char8 y);
short4 __builtin_vmp_vrnd0_16(short4 x, short4 y);
int2 __builtin_vmp_vrnd0_32(int2 x, int2 y);
uchar8 __builtin_vmp_vrnd0_u8(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vrnd0_u16(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vrnd0_u32(uint2 x, uint2 y);

//The following builtins perform a V_RND00 instruction on the vector data path in various sizes.
char8 __builtin_vmp_vrnd00_8(char8 x, char8 y);
short4 __builtin_vmp_vrnd00_16(short4 x, short4 y);
int2 __builtin_vmp_vrnd00_32(int2 x, int2 y);
uchar8 __builtin_vmp_vrnd00_u8(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vrnd00_u16(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vrnd00_u32(uint2 x, uint2 y);

//The following builtins perform a V_RND instruction on the vector data path in various sizes.
char8 __builtin_vmp_vrnd_8(char8 x, char8 y);
short4 __builtin_vmp_vrnd_16(short4 x, short4 y);
int2 __builtin_vmp_vrnd_32(int2 x, int2 y);
uchar8 __builtin_vmp_vrnd_u8(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vrnd_u16(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vrnd_u32(uint2 x, uint2 y);

//The following builtins perform a V_RNDRC instruction on the vector data path in various sizes.
char8 __builtin_vmp_vrndrc_8(char8 x, char8 y, int rc);
short4 __builtin_vmp_vrndrc_16(short4 x, short4 y, int rc);
int2 __builtin_vmp_vrndrc_32(int2 x, int2 y, int rc);
uchar8 __builtin_vmp_vrndrc_u8(uchar8 x, uchar8 y, int rc);
ushort4 __builtin_vmp_vrndrc_u16(ushort4 x, ushort4 y, int rc);
uint2 __builtin_vmp_vrndrc_u32(uint2 x, uint2 y, int rc);

//The following builtins perform a V_MIXL instruction on the vector data path in various sizes.
char8 __builtin_vmp_vmixl_8(char8 x, char8 y);
short4 __builtin_vmp_vmixl_16(short4 x, short4 y);
int2 __builtin_vmp_vmixl_32(int2 x, int2 y);

//The following builtins perform a V_MIXR instruction on the vector data path in various sizes.
char8 __builtin_vmp_vmixr_8(char8 x, char8 y);
short4 __builtin_vmp_vmixr_16(short4 x, short4 y);
int2 __builtin_vmp_vmixr_32(int2 x, int2 y);

//The following builtins perform a V_PERM instruction on the vector data path in various sizes. The
//scalar permutation mask can be computed by the builtin __builtin_vmp_vperm_value_X.
char8 __builtin_vmp_vperm_8(char8 x, uchar8 mask);
short4 __builtin_vmp_vperm_16(short4 x, ushort4 mask);
int2 __builtin_vmp_vperm_32(int2 x, uint2 mask);

//The following builtins perform a V_PERMREG instruction on the vector data path in various sizes. The
//scalar permutation mask can be computed by the builtin __builtin_vmp_vperm_value_X.
char8 __builtin_vmp_vpermreg_8(char8 x, char8 y, uchar8 mask);
short4 __builtin_vmp_vpermreg_16(short4 x, short4 y, ushort4 mask);
int2 __builtin_vmp_vpermreg_32(int2 x, int2 y, uint2 mask);

//The following builtins perform a V_PERMREG2 instruction on the vector data path in various sizes. The
//scalar permutation mask can be computed by the builtin __builtin_vmp_vperm_value_X.
char8 __builtin_vmp_vpermreg2_8(char8 x, char8 y, uchar8 mask);
short4 __builtin_vmp_vpermreg2_16(short4 x, short4 y, ushort4 mask);
int2 __builtin_vmp_vpermreg2_32(int2 x, int2 y, uint2 mask);

//The following builtins perform a V_MIN instruction on the vector data path in various sizes.
char8 __builtin_vmp_vmin_8(char8 x, char8 y);
short4 __builtin_vmp_vmin_16(short4 x, short4 y);
int2 __builtin_vmp_vmin_32(int2 x, int2 y);
uchar8 __builtin_vmp_vmin_u8(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vmin_u16(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vmin_u32(uint2 x, uint2 y);

//The following builtins perform a V_MAX instruction on the vector data path in various sizes.
char8 __builtin_vmp_vmax_8(char8 x, char8 y);
short4 __builtin_vmp_vmax_16(short4 x, short4 y);
int2 __builtin_vmp_vmax_32(int2 x, int2 y);
uchar8 __builtin_vmp_vmax_u8(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vmax_u16(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vmax_u32(uint2 x, uint2 y);

//The following builtins perform a V_CLIP instruction on the vector data path in various sizes.
char8 __builtin_vmp_vclip_8(char8 x, char8 y);
short4 __builtin_vmp_vclip_16(short4 x, short4 y);
int2 __builtin_vmp_vclip_32(int2 x, int2 y);

//The following builtins perform a V_CLIP2 instruction on the vector data path in various sizes.
char8 __builtin_vmp_vclip2_8(char8 x, char8 y);
short4 __builtin_vmp_vclip2_16(short4 x, short4 y);
int2 __builtin_vmp_vclip2_32(int2 x, int2 y);

//The following builtins perform a V_ABSADD instruction on the vector data path in various sizes.
char8 __builtin_vmp_vabsadd_8(char8 x, char8 y);
short4 __builtin_vmp_vabsadd_16(short4 x, short4 y);
int2 __builtin_vmp_vabsadd_32(int2 x, int2 y);
uchar8 __builtin_vmp_vabsadd_u8(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vabsadd_u16(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vabsadd_u32(uint2 x, uint2 y);

//The following builtins perform a V_AVG instruction on the vector data path in various sizes.
char8 __builtin_vmp_vavg_8(char8 x, char8 y);
short4 __builtin_vmp_vavg_16(short4 x, short4 y);
int2 __builtin_vmp_vavg_32(int2 x, int2 y);
uchar8 __builtin_vmp_vavg_u8(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vavg_u16(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vavg_u32(uint2 x, uint2 y);

//The following builtins perform a V_AVGRC instruction on the vector data path in various sizes.
char8 __builtin_vmp_vavgrc_8(char8 x, char8 y, int rc);
short4 __builtin_vmp_vavgrc_16(short4 x, short4 y, int rc);
int2 __builtin_vmp_vavgrc_32(int2 x, int2 y, int rc);
uchar8 __builtin_vmp_vavgrc_u8(uchar8 x, uchar8 y, int rc);
ushort4 __builtin_vmp_vavgrc_u16(ushort4 x, ushort4 y, int rc);
uint2 __builtin_vmp_vavgrc_u32(uint2 x, uint2 y, int rc);

//The following builtins perform a V_ADDEXPNDL instruction on the vector data path in various sizes.
short4 __builtin_vmp_vaddexpndl_8(char8 x, char8 y);
int2 __builtin_vmp_vaddexpndl_16(short4 x, short4 y);
long1 __builtin_vmp_vaddexpndl_32(int2 x, int2 y);
ushort4 __builtin_vmp_vaddexpndl_u8(uchar8 x, uchar8 y);
uint2 __builtin_vmp_vaddexpndl_u16(ushort4 x, ushort4 y);
ulong1 __builtin_vmp_vaddexpndl_u32(uint2 x, uint2 y);

//The following builtins perform a V_ADDEXPNDH instruction on the vector data path in various sizes.
short4 __builtin_vmp_vaddexpndh_8(char8 x, char8 y);
int2 __builtin_vmp_vaddexpndh_16(short4 x, short4 y);
long1 __builtin_vmp_vaddexpndh_32(int2 x, int2 y);
ushort4 __builtin_vmp_vaddexpndh_u8(uchar8 x, uchar8 y);
uint2 __builtin_vmp_vaddexpndh_u16(ushort4 x, ushort4 y);
ulong1 __builtin_vmp_vaddexpndh_u32(uint2 x, uint2 y);

//The following builtins perform a mul hi instruction on the vector data path in various sizes.
char8 __builtin_vmp_vmulhi_8(char8 x, char8 y);
short4 __builtin_vmp_vmulhi_16(short4 x, short4 y);
int2 __builtin_vmp_vmulhi_32(int2 x, int2 y);
uchar8 __builtin_vmp_vmulhi_u8(uchar8 x, uchar8 y);
ushort4 __builtin_vmp_vmulhi_u16(ushort4 x, ushort4 y);
uint2 __builtin_vmp_vmulhi_u32(uint2 x, uint2 y);

//The following builtins perform an unsigned mul instruction on 64 Bit data.
ulong1 __builtin_vmp_vmul_u64(ulong1 x, ulong1 y);


//The following builtins get the results of a vector comparison as argument and return 1 if the most
//significant bit in all components is set; otherwise it returns 0.
int __builtin_vmp_all_8(char8 x, uchar8 mask);
int __builtin_vmp_all_16(short4 x, ushort4 mask);
int __builtin_vmp_all_32(int2 x, uint2 mask);
int __builtin_vmp_all_u8(uchar8 x, uchar8 mask);
int __builtin_vmp_all_u16(ushort4 x, ushort4 mask);
int __builtin_vmp_all_u32(uint2 x, uint2 mask);

//The following builtins get the results of a vector comparison as argument and return 1 if the most
//significant bit in any component is set; otherwise it returns 0.
int __builtin_vmp_any_8(char8 x, uchar8 mask);
int __builtin_vmp_any_16(short4 x, ushort4 mask);
int __builtin_vmp_any_32(int2 x, uint2 mask);
int __builtin_vmp_any_u8(uchar8 x, uchar8 mask);
int __builtin_vmp_any_u16(ushort4 x, ushort4 mask);
int __builtin_vmp_any_u32(uint2 x, uint2 mask);

//The following builtins return the number of leading zero bits in the argument.
uchar8 __builtin_vmp_clz_8(uchar8 x);
ushort4 __builtin_vmp_clz_16(ushort4 x);
uint2 __builtin_vmp_clz_32(uint2 x);

//The following builtins return the number of trailing zero bits in the argument.
uchar8 __builtin_vmp_ctz_8(uchar8 x);
ushort4 __builtin_vmp_ctz_16(ushort4 x);
uint2 __builtin_vmp_ctz_32(uint2 x);

//The following builtins return the number of non-zero bits in the argument.
uchar8 __builtin_vmp_popcount_8(uchar8 x);
ushort4 __builtin_vmp_popcount_16(ushort4 x);
uint2 __builtin_vmp_popcount_32(uint2 x);

//The following builtins return shuffeled version of the input vectors
char8 builtin_shuffle8(char8 a, char8 b, int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7);
short4 builtin_shuffle4(short4 a, short4 b, int s0, int s1, int s2, int s3);
int2 builtin_shuffle2(int2 a, int2 b, int s0, int s1);

uchar8 builtin_shuffle8(uchar8 a, uchar8 b, int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7);
ushort4 builtin_shuffle4(ushort4 a, ushort4 b, int s0, int s1, int s2, int s3);
uint2 builtin_shuffle2(uint2 a, uint2 b, int s0, int s1);

// The following builtins map to user defined macro calls
uint __builtin_vmp_user_defined_0(uint p0, uint p1, uint p2, uint p3, uint p4,
                                  uint p5, uint2 p6, uint2 p7, uint2 p8,
                                  uint2 p9, uint2 p10, uint2 p11);

uint __builtin_vmp_user_defined_1(uint p0, uint p1, uint p2, uint p3, uint p4,
                                  uint p5, uint2 p6, uint2 p7, uint2 p8,
                                  uint2 p9, uint2 p10, uint2 p11);

uint __builtin_vmp_user_defined_2(uint p0, uint p1, uint p2, uint p3, uint p4,
                                  uint p5, uint2 p6, uint2 p7, uint2 p8,
                                  uint2 p9, uint2 p10, uint2 p11);

uint __builtin_vmp_user_defined_3(uint p0, uint p1, uint p2, uint p3, uint p4,
                                  uint p5, uint2 p6, uint2 p7, uint2 p8,
                                  uint2 p9, uint2 p10, uint2 p11);

uint __builtin_vmp_user_defined_4(uint p0, uint p1, uint p2, uint p3, uint p4,
                                  uint p5, uint2 p6, uint2 p7, uint2 p8,
                                  uint2 p9, uint2 p10, uint2 p11);

uint __builtin_vmp_user_defined_5(uint p0, uint p1, uint p2, uint p3, uint p4,
                                  uint p5, uint2 p6, uint2 p7, uint2 p8,
                                  uint2 p9, uint2 p10, uint2 p11);

uint __builtin_vmp_user_defined_6(uint p0, uint p1, uint p2, uint p3, uint p4,
                                  uint p5, uint2 p6, uint2 p7, uint2 p8,
                                  uint2 p9, uint2 p10, uint2 p11);

uint __builtin_vmp_user_defined_7(uint p0, uint p1, uint p2, uint p3, uint p4,
                                  uint p5, uint2 p6, uint2 p7, uint2 p8,
                                  uint2 p9, uint2 p10, uint2 p11);

uint2 __builtin_vmp_user_defined_8(uint p0, uint p1, uint p2, uint p3, uint p4,
                                   uint p5, uint2 p6, uint2 p7, uint2 p8,
                                   uint2 p9, uint2 p10, uint2 p11);

uint2 __builtin_vmp_user_defined_9(uint p0, uint p1, uint p2, uint p3, uint p4,
                                   uint p5, uint2 p6, uint2 p7, uint2 p8,
                                   uint2 p9, uint2 p10, uint2 p11);

uint2 __builtin_vmp_user_defined_10(uint p0, uint p1, uint p2, uint p3, uint p4,
                                    uint p5, uint2 p6, uint2 p7, uint2 p8,
                                    uint2 p9, uint2 p10, uint2 p11);

uint2 __builtin_vmp_user_defined_11(uint p0, uint p1, uint p2, uint p3, uint p4,
                                    uint p5, uint2 p6, uint2 p7, uint2 p8,
                                    uint2 p9, uint2 p10, uint2 p11);

uint2 __builtin_vmp_user_defined_12(uint p0, uint p1, uint p2, uint p3, uint p4,
                                    uint p5, uint2 p6, uint2 p7, uint2 p8,
                                    uint2 p9, uint2 p10, uint2 p11);

uint2 __builtin_vmp_user_defined_13(uint p0, uint p1, uint p2, uint p3, uint p4,
                                    uint p5, uint2 p6, uint2 p7, uint2 p8,
                                    uint2 p9, uint2 p10, uint2 p11);

uint2 __builtin_vmp_user_defined_14(uint p0, uint p1, uint p2, uint p3, uint p4,
                                    uint p5, uint2 p6, uint2 p7, uint2 p8,
                                    uint2 p9, uint2 p10, uint2 p11);

uint2 __builtin_vmp_user_defined_15(uint p0, uint p1, uint p2, uint p3, uint p4,
                                    uint p5, uint2 p6, uint2 p7, uint2 p8,
                                    uint2 p9, uint2 p10, uint2 p11);

void __builtin_vmp_write_BIU_DMA_CTRL(uint flags);

char8 __builtin_vmp_bitcast_8(char8 v0);
char8 __builtin_vmp_bitcast_8(uchar8 v0);
char8 __builtin_vmp_bitcast_8(short4 v0);
char8 __builtin_vmp_bitcast_8(ushort4 v0);
char8 __builtin_vmp_bitcast_8(int2 v0);
char8 __builtin_vmp_bitcast_8(uint2 v0);
char8 __builtin_vmp_bitcast_8(long1 v0);
char8 __builtin_vmp_bitcast_8(ulong1 v0);

short4 __builtin_vmp_bitcast_16(char8 v0);
short4 __builtin_vmp_bitcast_16(uchar8 v0);
short4 __builtin_vmp_bitcast_16(short4 v0);
short4 __builtin_vmp_bitcast_16(ushort4 v0);
short4 __builtin_vmp_bitcast_16(int2 v0);
short4 __builtin_vmp_bitcast_16(uint2 v0);
short4 __builtin_vmp_bitcast_16(long1 v0);
short4 __builtin_vmp_bitcast_16(ulong1 v0);

int2 __builtin_vmp_bitcast_32(char8 v0);
int2 __builtin_vmp_bitcast_32(uchar8 v0);
int2 __builtin_vmp_bitcast_32(short4 v0);
int2 __builtin_vmp_bitcast_32(ushort4 v0);
int2 __builtin_vmp_bitcast_32(int2 v0);
int2 __builtin_vmp_bitcast_32(uint2 v0);
int2 __builtin_vmp_bitcast_32(long1 v0);
int2 __builtin_vmp_bitcast_32(ulong1 v0);

long1 __builtin_vmp_bitcast_64(char8 v0);
long1 __builtin_vmp_bitcast_64(uchar8 v0);
long1 __builtin_vmp_bitcast_64(short4 v0);
long1 __builtin_vmp_bitcast_64(ushort4 v0);
long1 __builtin_vmp_bitcast_64(int2 v0);
long1 __builtin_vmp_bitcast_64(uint2 v0);
long1 __builtin_vmp_bitcast_64(long1 v0);
long1 __builtin_vmp_bitcast_64(ulong1 v0);

short8 __builtin_vmp_bitcast_16x2(short8 v0);
short8 __builtin_vmp_bitcast_16x2(ushort8 v0);
short8 __builtin_vmp_bitcast_16x2(int4 v0);
short8 __builtin_vmp_bitcast_16x2(uint4 v0);
short8 __builtin_vmp_bitcast_16x2(long2 v0);
short8 __builtin_vmp_bitcast_16x2(ulong2 v0);

int4 __builtin_vmp_bitcast_32x2(short8 v0);
int4 __builtin_vmp_bitcast_32x2(ushort8 v0);
int4 __builtin_vmp_bitcast_32x2(int4 v0);
int4 __builtin_vmp_bitcast_32x2(uint4 v0);
int4 __builtin_vmp_bitcast_32x2(long2 v0);
int4 __builtin_vmp_bitcast_32x2(ulong2 v0);

long2 __builtin_vmp_bitcast_64x2(short8 v0);
long2 __builtin_vmp_bitcast_64x2(ushort8 v0);
long2 __builtin_vmp_bitcast_64x2(int4 v0);
long2 __builtin_vmp_bitcast_64x2(uint4 v0);
long2 __builtin_vmp_bitcast_64x2(long2 v0);
long2 __builtin_vmp_bitcast_64x2(ulong2 v0);


char8 __builtin_vmp_vselect_8(char8 f, char8 t, uchar8 s);
short4 __builtin_vmp_vselect_16(short4 f, short4 t, ushort4 s);
int2 __builtin_vmp_vselect_32(int2 f, int2 t, uint2 s);


// special case for oclint to avoid incomplete types for .s00 in class uint2
uint2 as_uint2(uint v0);

inline char8 as_char8 (char8 v0);
inline char8 as_char8 (uchar8 v0);
inline char8 as_char8 (short4 v0);
inline char8 as_char8 (ushort4 v0);
inline char8 as_char8 (int2 v0);
inline char8 as_char8 (uint2 v0);
inline char8 as_char8 (long1 v0);
inline char8 as_char8 (ulong1 v0);
inline uchar8 as_uchar8 (char8 v0);
inline uchar8 as_uchar8 (uchar8 v0);
inline uchar8 as_uchar8 (short4 v0);
inline uchar8 as_uchar8 (ushort4 v0);
inline uchar8 as_uchar8 (int2 v0);
inline uchar8 as_uchar8 (uint2 v0);
inline uchar8 as_uchar8 (long1 v0);
inline uchar8 as_uchar8 (ulong1 v0);
inline short4 as_short4 (char8 v0);
inline short4 as_short4 (uchar8 v0);
inline short4 as_short4 (short4 v0);
inline short4 as_short4 (ushort4 v0);
inline short4 as_short4 (int2 v0);
inline short4 as_short4 (uint2 v0);
inline short4 as_short4 (long1 v0);
inline short4 as_short4 (ulong1 v0);
inline ushort4 as_ushort4 (char8 v0);
inline ushort4 as_ushort4 (uchar8 v0);
inline ushort4 as_ushort4 (short4 v0);
inline ushort4 as_ushort4 (ushort4 v0);
inline ushort4 as_ushort4 (int2 v0);
inline ushort4 as_ushort4 (uint2 v0);
inline ushort4 as_ushort4 (long1 v0);
inline ushort4 as_ushort4 (ulong1 v0);
inline int2 as_int2 (char8 v0);
inline int2 as_int2 (uchar8 v0);
inline int2 as_int2 (short4 v0);
inline int2 as_int2 (ushort4 v0);
inline int2 as_int2 (int2 v0);
inline int2 as_int2 (uint2 v0);
inline int2 as_int2 (long1 v0);
inline int2 as_int2 (ulong1 v0);
inline uint2 as_uint2 (char8 v0);
inline uint2 as_uint2 (uchar8 v0);
inline uint2 as_uint2 (short4 v0);
inline uint2 as_uint2 (ushort4 v0);
inline uint2 as_uint2 (int2 v0);
inline uint2 as_uint2 (uint2 v0);
inline uint2 as_uint2 (long1 v0);
inline uint2 as_uint2 (ulong1 v0);
inline long1 as_long1 (char8 v0);
inline long1 as_long1 (uchar8 v0);
inline long1 as_long1 (short4 v0);
inline long1 as_long1 (ushort4 v0);
inline long1 as_long1 (int2 v0);
inline long1 as_long1 (uint2 v0);
inline long1 as_long1 (long1 v0);
inline long1 as_long1 (ulong1 v0);
inline ulong1 as_ulong1 (char8 v0);
inline ulong1 as_ulong1 (uchar8 v0);
inline ulong1 as_ulong1 (short4 v0);
inline ulong1 as_ulong1 (ushort4 v0);
inline ulong1 as_ulong1 (int2 v0);
inline ulong1 as_ulong1 (uint2 v0);
inline ulong1 as_ulong1 (long1 v0);
inline ulong1 as_ulong1 (ulong1 v0);
inline short8 as_short8 (short8 v0);
inline short8 as_short8 (ushort8 v0);
inline short8 as_short8 (int4 v0);
inline short8 as_short8 (uint4 v0);
inline short8 as_short8 (long2 v0);
inline short8 as_short8 (ulong2 v0);
inline ushort8 as_ushort8 (short8 v0);
inline ushort8 as_ushort8 (ushort8 v0);
inline ushort8 as_ushort8 (int4 v0);
inline ushort8 as_ushort8 (uint4 v0);
inline ushort8 as_ushort8 (long2 v0);
inline ushort8 as_ushort8 (ulong2 v0);
inline int4 as_int4 (short8 v0);
inline int4 as_int4 (ushort8 v0);
inline int4 as_int4 (int4 v0);
inline int4 as_int4 (uint4 v0);
inline int4 as_int4 (long2 v0);
inline int4 as_int4 (ulong2 v0);
inline uint4 as_uint4 (short8 v0);
inline uint4 as_uint4 (ushort8 v0);
inline uint4 as_uint4 (int4 v0);
inline uint4 as_uint4 (uint4 v0);
inline uint4 as_uint4 (long2 v0);
inline uint4 as_uint4 (ulong2 v0);
inline long2 as_long2 (short8 v0);
inline long2 as_long2 (ushort8 v0);
inline long2 as_long2 (int4 v0);
inline long2 as_long2 (uint4 v0);
inline long2 as_long2 (long2 v0);
inline long2 as_long2 (ulong2 v0);
inline ulong2 as_ulong2 (short8 v0);
inline ulong2 as_ulong2 (ushort8 v0);
inline ulong2 as_ulong2 (int4 v0);
inline ulong2 as_ulong2 (uint4 v0);
inline ulong2 as_ulong2 (long2 v0);
inline ulong2 as_ulong2 (ulong2 v0);
inline uchar8 vabs (char8 x);
inline uchar8 vabs (uchar8 x);
inline ushort4 vabs (short4 x);
inline ushort4 vabs (ushort4 x);
inline uint2 vabs (int2 x);
inline uint2 vabs (uint2 x);
inline uchar8 abs_diff (char8 x, char8 y);
inline uchar8 abs_diff (uchar8 x, uchar8 y);
inline ushort4 abs_diff (short4 x, short4 y);
inline ushort4 abs_diff (ushort4 x, ushort4 y);
inline uint2 abs_diff (int2 x, int2 y);
inline uint2 abs_diff (uint2 x, uint2 y);
inline char8 add_sat (char8 x, char8 y);
inline uchar8 add_sat (uchar8 x, uchar8 y);
inline short4 add_sat (short4 x, short4 y);
inline ushort4 add_sat (ushort4 x, ushort4 y);
inline int2 add_sat (int2 x, int2 y);
inline uint2 add_sat (uint2 x, uint2 y);
inline char8 hadd (char8 x, char8 y);
inline uchar8 hadd (uchar8 x, uchar8 y);
inline short4 hadd (short4 x, short4 y);
inline ushort4 hadd (ushort4 x, ushort4 y);
inline int2 hadd (int2 x, int2 y);
inline uint2 hadd (uint2 x, uint2 y);
inline char8 rhadd (char8 x, char8 y);
inline uchar8 rhadd (uchar8 x, uchar8 y);
inline short4 rhadd (short4 x, short4 y);
inline ushort4 rhadd (ushort4 x, ushort4 y);
inline int2 rhadd (int2 x, int2 y);
inline uint2 rhadd (uint2 x, uint2 y);
inline char8 clamp (char8 x, char8 minval, char8 maxval);
inline short4 clamp (short4 x, short4 minval, short4 maxval);
inline int2 clamp (int2 x, int2 minval, int2 maxval);
inline uchar8 clamp (uchar8 x, uchar8 minval, uchar8 maxval);
inline ushort4 clamp (ushort4 x, ushort4 minval, ushort4 maxval);
inline uint2 clamp (uint2 x, uint2 minval, uint2 maxval);

inline char8 clz (char8 x);
inline uchar8 clz (uchar8 x);
inline short4 clz (short4 x);
inline ushort4 clz (ushort4 x);
inline int2 clz (int2 x);
inline uint2 clz (uint2 x);
inline char8 ctz (char8 x);
inline uchar8 ctz (uchar8 x);
inline short4 ctz (short4 x);
inline ushort4 ctz (ushort4 x);
inline int2 ctz (int2 x);
inline uint2 ctz (uint2 x);
inline char8 mul_hi (char8 x, char8 y);
inline uchar8 mul_hi (uchar8 x, uchar8 y);
inline short4 mul_hi (short4 x, short4 y);
inline ushort4 mul_hi (ushort4 x, ushort4 y);
inline int2 mul_hi (int2 x, int2 y);
inline uint2 mul_hi (uint2 x, uint2 y);
inline char8 mad_hi (char8 a, char8 b, char8 c);
inline uchar8 mad_hi (uchar8 a, uchar8 b, uchar8 c);
inline short4 mad_hi (short4 a, short4 b, short4 c);
inline ushort4 mad_hi (ushort4 a, ushort4 b, ushort4 c);
inline int2 mad_hi (int2 a, int2 b, int2 c);
inline uint2 mad_hi (uint2 a, uint2 b, uint2 c);
inline short8 mad_sat (char8 a, char8 b, short8 c);
inline ushort8 mad_sat (uchar8 a, uchar8 b, ushort8 c);
inline int4 mad_sat (short4 a, short4 b, int4 c);
inline uint4 mad_sat (ushort4 a, ushort4 b, uint4 c);
inline short8 mad (char8 a, char8 b, short8 c);
inline ushort8 mad (uchar8 a, uchar8 b, ushort8 c);
inline int4 mad (short4 a, short4 b, int4 c);
inline uint4 mad (ushort4 a, ushort4 b, uint4 c);
inline long2 mad (int2 a, int2 b, long2 c);
inline ulong2 mad (uint2 a, uint2 b, ulong2 c);
inline char8 max (char8 x, char8 y);
inline uchar8 max (uchar8 x, uchar8 y);
inline short4 max (short4 x, short4 y);
inline ushort4 max (ushort4 x, ushort4 y);
inline int2 max (int2 x, int2 y);
inline uint2 max (uint2 x, uint2 y);
inline char8 min (char8 x, char8 y);
inline uchar8 min (uchar8 x, uchar8 y);
inline short4 min (short4 x, short4 y);
inline ushort4 min (ushort4 x, ushort4 y);
inline int2 min (int2 x, int2 y);
inline uint2 min (uint2 x, uint2 y);
inline char8 rotate (char8 x, char8 i);
inline uchar8 rotate (uchar8 x, uchar8 i);
inline short4 rotate (short4 x, short4 i);
inline ushort4 rotate (ushort4 x, ushort4 i);
inline int2 rotate (int2 x, int2 i);
inline uint2 rotate (uint2 x, uint2 i);
inline char8 sub_sat (char8 x, char8 y);
inline uchar8 sub_sat (uchar8 x, uchar8 y);
inline short4 sub_sat (short4 x, short4 y);
inline ushort4 sub_sat (ushort4 x, ushort4 y);
inline int2 sub_sat (int2 x, int2 y);
inline uint2 sub_sat (uint2 x, uint2 y);
inline short8 upsample (char8 hi, uchar8 lo);
inline ushort8 upsample(uchar8 hi, uchar8 lo);
inline int4 upsample (short4 hi, ushort4 lo);
inline uint4 upsample (ushort4 hi, ushort4 lo);
inline long2 upsample (int2 hi, uint2 lo);
inline ulong2 upsample (uint2 hi, uint2 lo);
inline char8 popcount (char8 x);
inline uchar8 popcount (uchar8 x);
inline short4 popcount (short4 x);
inline ushort4 popcount (ushort4 x);
inline int2 popcount (int2 x);
inline uint2 popcount (uint2 x);
inline char8 isequal (char8 v0, char8 v1);
inline short4 isequal (short4 v0, short4 v1);
inline int2 isequal (int2 v0, int2 v1);
inline char8 isequal (uchar8 v0, uchar8 v1);
inline short4 isequal (ushort4 v0, ushort4 v1);
inline int2 isequal (uint2 v0, uint2 v1);
inline char8 isnotequal (char8 v0, char8 v1);
inline short4 isnotequal (short4 v0, short4 v1);
inline int2 isnotequal (int2 v0, int2 v1);
inline char8 isnotequal (uchar8 v0, uchar8 v1);
inline short4 isnotequal (ushort4 v0, ushort4 v1);
inline int2 isnotequal (uint2 v0, uint2 v1);
inline char8 isgreater (char8 v0, char8 v1);
inline short4 isgreater (short4 v0, short4 v1);
inline int2 isgreater (int2 v0, int2 v1);
inline char8 isgreater (uchar8 v0, uchar8 v1);
inline short4 isgreater (ushort4 v0, ushort4 v1);
inline int2 isgreater (uint2 v0, uint2 v1);
inline char8 isgreaterequal (char8 v0, char8 v1);
inline short4 isgreaterequal (short4 v0, short4 v1);
inline int2 isgreaterequal (int2 v0, int2 v1);
inline char8 isgreaterequal (uchar8 v0, uchar8 v1);
inline short4 isgreaterequal (ushort4 v0, ushort4 v1);
inline int2 isgreaterequal (uint2 v0, uint2 v1);
inline char8 isless (char8 v0, char8 v1);
inline short4 isless (short4 v0, short4 v1);
inline int2 isless (int2 v0, int2 v1);
inline char8 isless (uchar8 v0, uchar8 v1);
inline short4 isless (ushort4 v0, ushort4 v1);
inline int2 isless (uint2 v0, uint2 v1);
inline char8 islessequal (char8 v0, char8 v1);
inline short4 islessequal (short4 v0, short4 v1);
inline int2 islessequal (int2 v0, int2 v1);
inline char8 islessequal (uchar8 v0, uchar8 v1);
inline short4 islessequal (ushort4 v0, ushort4 v1);
inline int2 islessequal (uint2 v0, uint2 v1);
inline char8 signbit (char8 a);
inline short4 signbit (short4 a);
inline int2 signbit (int2 a);
inline int any_masked (char8 v0, uchar8 mask);
inline int any_masked (short4 v0, ushort4 mask);
inline int any_masked (int2 v0, uint2 mask);
inline int any_masked (uchar8 v0, uchar8 mask);
inline int any_masked (ushort4 v0, ushort4 mask);
inline int any_masked (uint2 v0, uint2 mask);
inline int any (char8 v0);
inline int any (short4 v0);
inline int any (int2 v0);
inline int any (uchar8 v0);
inline int any (ushort4 v0);
inline int any (uint2 v0);
inline int all_masked (char8 v0, uchar8 mask);
inline int all_masked (short4 v0, ushort4 mask);
inline int all_masked (int2 v0, uint2 mask);
inline int all_masked (uchar8 v0, uchar8 mask);
inline int all_masked (ushort4 v0, ushort4 mask);
inline int all_masked (uint2 v0, uint2 mask);
inline int all (char8 v0);
inline int all (short4 v0);
inline int all (int2 v0);
inline int all (uchar8 v0);
inline int all (ushort4 v0);
inline int all (uint2 v0);
inline char8 bitselect (char8 a, char8 b, char8 c);
inline short4 bitselect (short4 a, short4 b, short4 c);
inline int2 bitselect (int2 a, int2 b, int2 c);
inline uchar8 bitselect (uchar8 a, uchar8 b, uchar8 c);
inline ushort4 bitselect (ushort4 a, ushort4 b, ushort4 c);
inline uint2 bitselect (uint2 a, uint2 b, uint2 c);


inline char8 select (char8 f, char8 t, uchar8 s);
inline uchar8 select (uchar8 f, uchar8 t, uchar8 s);
inline short4 select (short4 f, short4 t, ushort4 s);
inline ushort4 select (ushort4 f, ushort4 t, ushort4 s);
inline int2 select (int2 f, int2 t, uint2 s);
inline uint2 select (uint2 f, uint2 t, uint2 s);
inline char8 select (char8 f, char8 t, char8 s);
inline uchar8 select (uchar8 f, uchar8 t, char8 s);
inline short4 select (short4 f, short4 t, short4 s);
inline ushort4 select (ushort4 f, ushort4 t, short4 s);
inline int2 select (int2 f, int2 t, int2 s);
inline uint2 select (uint2 f, uint2 t, int2 s);


inline char8 shuffle (char8 vector1, uchar8 mask);
inline uchar8 shuffle (uchar8 vector1, uchar8 mask);
inline short4 shuffle (short4 vector1, ushort4 mask);
inline ushort4 shuffle (ushort4 vector1, ushort4 mask);
inline int2 shuffle (int2 vector1, uint2 mask);
inline uint2 shuffle (uint2 vector1, uint2 mask);

inline char8 shuffle2 (char8 vector1, char8 vector2, uchar8 mask);
inline uchar8 shuffle2 (uchar8 vector1, uchar8 vector2, uchar8 mask);
inline short4 shuffle2 (short4 vector1, short4 vector2, ushort4 mask);
inline uchar8 shuffle2 (uchar8 vector1, uchar8 vector2, uchar8 mask);
inline ushort4 shuffle2 (ushort4 vector1, ushort4 vector2, ushort4 mask);
inline int2 shuffle2 (int2 vector1, int2 vector2, uint2 mask);
inline uint2 shuffle2 (uint2 vector1, uint2 vector2, uint2 mask);

//lint -restore

#endif /* VID_VIDEANTISC_HPP_ */
