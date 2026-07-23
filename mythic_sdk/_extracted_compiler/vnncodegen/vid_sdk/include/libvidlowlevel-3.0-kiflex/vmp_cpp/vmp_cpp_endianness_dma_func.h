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
 * FILENAME: vmp_cpp_endianness_dma_func.h
 *
 * DESCRIPTION: videantis-C v-MP 4.x little endian DMA transfer functions
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @file
 * @brief videantis-C v-MP 4.x little endian DMA transfer functions
 *
 * This file provides always inline functions for little endian DMA transfer
 *
 */

#ifndef VMP_CPP_ENDIANNESS_DMA_FUNC
#define VMP_CPP_ENDIANNESS_DMA_FUNC

/* videantis lowlevel interface */
#include "vmp_cpp_dma.h"

// disable gnu-binary-literal warnings for the endianness DMA transfer functions
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-binary-literal"

/**
 * @brief Starts a little endian 64 write DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a little endian 64 write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param xferFlags_intWordaddr_lengthInt_length transfer flags (s3), internal word address (s2), length of segment internal 2D (s1) and length (s0)
 * @param reserved_strideInt_count3d_count reserved (s3), stride internal (s2), number of segments 3D (s1) and number of segments 2D (s0)
 * @param stride3d_stride stride 3D (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write_LE64(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    vmp_dma_descr.dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    vmp_dma_descr.dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    vmp_dma_descr.dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    vmp_dma_descr.dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_ENDIAN_MODE_LE64 | VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_WRITE | dmaChannelId);
}

/**
 * @brief Starts a linear little endian 64 write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear little endian 64 write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write_LE64(const uint2 extByteaddr,
                    __BYTEADDRESS uint2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write_LE64(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a little endian 32 write DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a little endian 32 write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param xferFlags_intWordaddr_lengthInt_length transfer flags (s3), internal word address (s2), length of segment internal 2D (s1) and length (s0)
 * @param reserved_strideInt_count3d_count reserved (s3), stride internal (s2), number of segments 3D (s1) and number of segments 2D (s0)
 * @param stride3d_stride stride 3D (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write_LE32(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    vmp_dma_descr.dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    vmp_dma_descr.dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    vmp_dma_descr.dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    vmp_dma_descr.dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_ENDIAN_MODE_LE32 | VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_WRITE | dmaChannelId);
}

/**
 * @brief Starts a linear little endian 32 write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear little endian 32 write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write_LE32(const uint2 extByteaddr,
                    __BYTEADDRESS uint2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write_LE32(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a little endian 16 write DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a little endian 16 write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param xferFlags_intWordaddr_lengthInt_length transfer flags (s3), internal word address (s2), length of segment internal 2D (s1) and length (s0)
 * @param reserved_strideInt_count3d_count reserved (s3), stride internal (s2), number of segments 3D (s1) and number of segments 2D (s0)
 * @param stride3d_stride stride 3D (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write_LE16(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    vmp_dma_descr.dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    vmp_dma_descr.dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    vmp_dma_descr.dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    vmp_dma_descr.dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_ENDIAN_MODE_LE16 | VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_WRITE | dmaChannelId);
}

/**
 * @brief Starts a linear little endian 16 write DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear little endian 16 write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_write_LE16(const uint2 extByteaddr,
                    __BYTEADDRESS uint2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_write_LE16(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a little endian 64 read DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a little endian 64 read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param xferFlags_intWordaddr_lengthInt_length transfer flags (s3), internal word address (s2), length of segment internal 2D (s1) and length (s0)
 * @param reserved_strideInt_count3d_count reserved (s3), stride internal (s2), number of segments 3D (s1) and number of segments 2D (s0)
 * @param stride3d_stride stride 3D (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read_LE64(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    vmp_dma_descr.dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    vmp_dma_descr.dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    vmp_dma_descr.dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    vmp_dma_descr.dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_ENDIAN_MODE_LE64 | VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_READ | dmaChannelId);
}

/**
 * @brief Starts a linear little endian 64 read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear little endian 64 read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
inline void vmp_dma_read_LE64(const uint2 extByteaddr,
                    __BYTEADDRESS uint2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read_LE64(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a little endian 32 read DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a little endian 32 read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param xferFlags_intWordaddr_lengthInt_length transfer flags (s3), internal word address (s2), length of segment internal 2D (s1) and length (s0)
 * @param reserved_strideInt_count3d_count reserved (s3), stride internal (s2), number of segments 3D (s1) and number of segments 2D (s0)
 * @param stride3d_stride stride 3D (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read_LE32(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    vmp_dma_descr.dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    vmp_dma_descr.dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    vmp_dma_descr.dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    vmp_dma_descr.dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_ENDIAN_MODE_LE32 | VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_READ | dmaChannelId);
}

/**
 * @brief Starts a linear little endian 32 read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear little endian 32 read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read_LE32(const uint2 extByteaddr,
                    __BYTEADDRESS uint2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read_LE32(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

/**
 * @brief Starts a little endian 16 read DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a little endian 16 read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param xferFlags_intWordaddr_lengthInt_length transfer flags (s3), internal word address (s2), length of segment internal 2D (s1) and length (s0)
 * @param reserved_strideInt_count3d_count reserved (s3), stride internal (s2), number of segments 3D (s1) and number of segments 2D (s0)
 * @param stride3d_stride stride 3D (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read_LE16(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    vmp_dma_descr.dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    vmp_dma_descr.dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    vmp_dma_descr.dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    vmp_dma_descr.dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_ENDIAN_MODE_LE16 | VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_READ | dmaChannelId);
}

/**
 * @brief Starts a linear little endian 16 read DMA transfer
 *
 * This function maps a simplified API call to the corresponding DMA descriptors of the channel and
 * starts a linear little endian 16 read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param intWordaddr internal word address
 * @param length length
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_dma_read_LE16(const uint2 extByteaddr,
                    __BYTEADDRESS  uint2* intWordaddr,
                    const uint length,
                    const uint dmaChannelId)
{
    vmp_dma_read_LE16(extByteaddr, as_ushort4(set_uint2(length, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)intWordaddr))),
                set_ushort4(0u), set_uint2(0u), dmaChannelId);
}

// restore clang diagnostics to previous diagnostic state
#pragma clang diagnostic pop

#endif // VMP_CPP_ENDIANNESS_DMA_FUNC
