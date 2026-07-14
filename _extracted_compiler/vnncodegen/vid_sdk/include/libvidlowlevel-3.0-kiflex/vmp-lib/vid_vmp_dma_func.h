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
 * FILENAME: vid_vmp_dma_func.h
 *
 * DESCRIPTION: videantis-C v-MP 4.x DMA functions
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @file
 * @brief videantis-C v-MP 4.x DMA functions
 *
 * This file provides always inline functions for simplified DMA
 * 1D and 2D transfers
 *
 */

#ifndef __VID_VMP_DMA_FUNC_H__
#define __VID_VMP_DMA_FUNC_H__


// vector data types
#include "vmp_cl/vmp_cl-types.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cl/vid_videantis-C.hpp"
#endif

// core DMA transfer functions
#include "vid_vmp_dma.h"

// avoid language incompatibility between OpenCL-C and C++ pointer sizes assumed by oclint
#ifdef __videantis_lint__
#define PCAST long
#else
#define PCAST int
#endif

// linear DMA functions
/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local void* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local void* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local uchar8* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local uchar8* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local char8* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local char8* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local ushort4* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local ushort4* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local short4* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local short4* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local uint2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local uint2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local int2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local int2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local ulong1* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local ulong1* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local long1* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a linear write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local long1* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

// 2d DMA functions
/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local void* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local void* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local uchar8* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local uchar8* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local char8* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local char8* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local ushort4* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local ushort4* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local short4* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local short4* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local uint2* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local uint2* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local int2* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local int2* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local ulong1* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local ulong1* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read(const uint2 extByteaddr,
                    __BYTEADDRESS __local long1* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_read(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

/**
 * @brief Starts a 2D write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a 2D write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param count_stride number of segments (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write(const uint2 extByteaddr,
                    __BYTEADDRESS __local long1* intWordaddr,
                    const uint length,
                    const uint2 count_stride,
                    const uint dmaChannelId)
{
    vmp_dma_write(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                as_ushort4(set_uint2(count_stride.s1, 0u)), set_uint2(count_stride.s0, 0u),
                dmaChannelId);
}

#endif // __VID_VMP_DMA_FUNC_H__
