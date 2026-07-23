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
* FILENAME: vmp_cpp-misra.h
*
* DESCRIPTION: interface to functions required for MISRA-C/C++
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef VMP_CPP_MISRA_H
#define VMP_CPP_MISRA_H

static constexpr long1 set_long1(long v)
{
    return long1{v};
}

static constexpr ulong1 set_ulong1(unsigned long v)
{
    return ulong1{v};
}

static constexpr long2 set_long2(long a, long b)
{
    return long2{a, b};
}

static constexpr long2 set_long2(long v)
{
    return long2{v, v};
}

static constexpr ulong2 set_ulong2(unsigned long a, unsigned long b)
{
    return ulong2{a, b};
}

static constexpr ulong2 set_ulong2(unsigned long v)
{
    return ulong2{v, v};
}

static constexpr int2 set_int2(int a, int b)
{
    return int2{a, b};
}

static constexpr int2 set_int2(int v)
{
    return int2{v, v};
}

static constexpr uint2 set_uint2(unsigned a, unsigned b)
{
    return uint2{a, b};
}

static constexpr uint2 set_uint2(unsigned v)
{
    return uint2{v, v};
}

static constexpr short4 set_short4(short a, short b, short c, short d)
{
    return short4{a, b, c, d};
}

static constexpr short4 set_short4(short v)
{
    return short4{v, v, v, v};
}

static constexpr ushort4 set_ushort4(unsigned short a, unsigned short b,
                                     unsigned short c, unsigned short d)
{
    return ushort4{a, b, c, d};
}

static constexpr ushort4 set_ushort4(unsigned short v)
{
    return ushort4{v, v, v, v};
}

static constexpr int4 set_int4(int a, int b, int c, int d)
{
    return int4{a, b, c, d};
}

static constexpr int4 set_int4(int v)
{
    return int4{v, v, v, v};
}

static constexpr uint4 set_uint4(unsigned a, unsigned b,
                                 unsigned c, unsigned d)
{
    return uint4{a, b, c, d};
}

static constexpr uint4 set_uint4(unsigned v)
{
    return uint4{v, v, v, v};
}

static constexpr short8 set_short8(short v)
{
    return short8{v, v, v, v, v, v, v, v};
}

static constexpr ushort8 set_ushort8(unsigned short v)
{
    return ushort8{v, v, v, v, v, v, v, v};
}

static constexpr char8 set_char8(char a, char b, char c, char d,
                                 char e, char f, char g, char h)
{
    return char8{a, b, c, d, e, f, g, h};
}

static constexpr char8 set_char8(char v)
{
    return char8{v, v, v, v, v, v, v, v};
}

static constexpr uchar8 set_uchar8(unsigned char a, unsigned char b,
                                   unsigned char c, unsigned char d,
                                   unsigned char e, unsigned char f,
                                   unsigned char g, unsigned char h)
{
    return uchar8{a, b, c, d, e, f, g, h};
}

static constexpr uchar8 set_uchar8(unsigned char v)
{
    return uchar8{v, v, v, v, v, v, v, v};
}

static constexpr char16 set_char16(char v)
{
    return char16{
        v, v, v, v, v, v, v, v,
        v, v, v, v, v, v, v, v};
}

static constexpr uchar16 set_uchar16(unsigned char v)
{
    return uchar16{
        v, v, v, v, v, v, v, v,
        v, v, v, v, v, v, v, v};
}

#if 0
template <typename V, typename T>
static constexpr V splat2(T v) { return V{v, v}; }

template <typename V, typename T>
static constexpr V splat4(T v) { return V{v, v, v, v}; }

template <typename V, typename T>
static constexpr V splat8(T v)
{
    return V{v, v, v, v, v, v, v, v};
}
#endif

#endif // VMP_CPP_MISRA_H
