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
 * FILENAME: vid_vmp_dma.h
 *
 * DESCRIPTION: videantis-C v-MP 4.x DMA definitions and generic functions
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @file
 * @brief videantis-C v-MP 4.x DMA definitions and generic functions
 *
 * This file provides definitions and always inline generic functions for DMA transfer
 *
 */

#ifndef __VID_VMP_DMA_H__
#define __VID_VMP_DMA_H__


// vector data types
#include "vmp_cl/vmp_cl-types.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cl/vid_videantis-C.hpp"
#endif

//lint -save

//lint -e146 Error 146: Assuming a binary constant
//lintReason: vmpcc compiler supports binary constants

//lint -e9026 Note 9026: Function-like macro, 'vmp_dma_wait_*', defined [MISRA 2012 Directive 4.9, advisory]
//lintReason Lint exceptions introduced because called builtin requires constant fold to an integer constant expression

// avoid language incompatibility between OpenCL-C and C++ assumed by oclint
#ifdef __videantis_lint__
#define VOLATILE
#else
#define VOLATILE volatile
#endif

/// number of DMA channels
#define VID_VMP_NUM_DMA_CHANNELS 16u
#if !defined(VMP_DMA_DESCR_ORG)
/// default base address of DMA descriptors
#define VMP_DMA_DESCR_ORG (0x1000u - 64u)
#endif

/// internal priority 0 for DMA transfer
#define VID_VMP_DMA_PRIO_INT0     (0b0000 << 8u)
/// internal priority 1 for DMA transfer
#define VID_VMP_DMA_PRIO_INT1     (0b0001 << 8u)
/// internal priority 2 for DMA transfer
#define VID_VMP_DMA_PRIO_INT2     (0b0010 << 8u)
/// internal priority 3 for DMA transfer
#define VID_VMP_DMA_PRIO_INT3     (0b0011 << 8u)
/// external priority 0 for DMA transfer
#define VID_VMP_DMA_PRIO_EXT0     (0b0000 << 8u)
/// external priority 1 for DMA transfer
#define VID_VMP_DMA_PRIO_EXT1     (0b0100 << 8u)
/// external priority 2 for DMA transfer
#define VID_VMP_DMA_PRIO_EXT2     (0b1000 << 8u)
/// external priority 3 for DMA transfer
#define VID_VMP_DMA_PRIO_EXT3     (0b1100 << 8u)
/// standard priority for DMA transfer
#define VID_VMP_DMA_PRIO_STD      0u
/// DMA transfer endian mode big endian 64 (default)
#define VID_VMP_DMA_ENDIAN_MODE_BE64 (0b00 << 6u)
/// DMA transfer endian mode little endian 64
#define VID_VMP_DMA_ENDIAN_MODE_LE64 (0b01 << 6u)
/// DMA transfer endian mode little endian 32
#define VID_VMP_DMA_ENDIAN_MODE_LE32 (0b10 << 6u)
/// DMA transfer endian mode little endian 16
#define VID_VMP_DMA_ENDIAN_MODE_LE16 (0b11 << 6u)
/// DMA memset transfer
#define VID_VMP_DMA_MEMSET        (1u << 5u)
/// DMA read or internal memset transfer
#define VID_VMP_DMA_READ          (1u << 4u)
/// DMA write or external memset transfer
#define VID_VMP_DMA_WRITE         (0u << 4u)

/**
 * @struct vmp_dma_t
 * @brief Structure contains DMA channel descriptor
 *
 * New large DMA descriptor layout, stored in DMEM/DMEM2/DMEM3, aligned to 64 word boundary
 *
 * | name             | byte offset | bits | description |
 * | ---------------- | ----------- | ---- | ----------- |
 * | extByteaddr      | 0x00..0x07  | 64   | External byte start address |
 * | xferFlags        | 0x08..0x08  | 16   | Transfer flags |
 * | intWordaddr      | 0x0A..0x0B  | 16   | Internal 64-bit word start address (IMEM starts at 0xF000) |
 * | segmentLengthInt | 0x0C..0x0D  | 10   | Length of segment for internal 2D scheme (0: internal 1D) Bit 0 will always be interpreted as zero, resulting in even values only |
 * | segmentLength    | 0x0E..0x0F  | 13   | Number of 64-bit words to be transferred per segment (0 is now defined and means ‘no transfer’) |
 * | (reserved)       | 0x10..0x11  | 16   | Reserved for future use, must be 0 |
 * | segmentStrideInt | 0x12..0x13  | 13   | Distance (number of 64-bit words) between the start address of two consecutive segments (only relevant for SEGMENT_LENGTH_INT>0). Bit 0 will always be interpreted as zero, resulting in even values only |
 * | segmentCount3d   | 0x14..0x15  | 12   | Number of segments to be transferred for 3rd dimension (0 means 1, i.e. 1D/2D) |
 * | segmentCount     | 0x16..0x17  | 12   | Number of segments to be transferred for 2nd dimension (0 is now defined and means 1, i.e. 1D) |
 * | segmentStride3d  | 0x18..0x1B  | 24   | Distance (number of 64-bit words) between the start address of two consecutive segments (only relevant for SEGMENT_COUNT_3D>1) |
 * | segmentStride    | 0x1C..0x1F  | 20   | Distance (number of 64-bit words) between the start address of two consecutive segments (only relevant for SEGMENT_COUNT>1) |
 *
 * Details of xferFlags: Bit position 0 / Bits 1 / Name NON_POSTED_WR<br>
 * Non-posted write flag (if supported by bus: transfer will not be indicated as finished until receiving write response), must be 0 for read/internal memset accesses
 *
 * @var uint2 vmp_dma_t::extByteaddr
 * Contains the external byte start address (s0 lower part of the address, s1 higher part of the address)
 * @var ushort4 vmp_dma_t::xferFlags_intWordaddr_lengthInt_length
 * Contains the transfer flags (s3), internal 64-bit word start address (s2), length of segment for internal 2D scheme (s1)
 * and number of 64-bit words to be transferred per segment (s0)
 * @var ushort4 vmp_dma_t::reserved_strideInt_count3d_count
 * Contains the reserved space for future use (s3), distance (number of 64-bit words) between the start address of two consecutive segments (s2),
 * number of segments to be transferred for 3rd dimension (s1) and number of segments to be transferred for 2nd dimension (s0)
 * @var uint2 vmp_dma_t::stride3d_stride
 * Contains the distance (number of 64-bit words) between the start address of two consecutive segments (s1) and
 * distance (number of 64-bit words) between the start address of two consecutive segments (s0)
 */
typedef struct {
    uint2 extByteaddr;
    ushort4 xferFlags_intWordaddr_lengthInt_length;
    ushort4 reserved_strideInt_count3d_count;
    uint2 stride3d_stride;
} vmp_dma_t;

/**
 * @struct vmp_dma_t
 * @brief Structure contains DMA descriptors for all channels
 *
 * New large DMA descriptors for all DMA channels
 *
 * @var vmp_dma_t vmp_dma_descr_t::dma_descr
 * Contains the DMA channel descriptor
 */

typedef struct {
    vmp_dma_t dma_descr[VID_VMP_NUM_DMA_CHANNELS];
} vmp_dma_descr_t;

/*
 * DMA generic functions
 */
/**
 * @brief Starts a read DMA transfer
 *
 * This function writes the input parameter to the corresponding DMA descriptors of the channel and
 * starts a read DMA transfer on the defined channel.
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
inline void vmp_dma_read(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    VOLATILE vmp_dma_descr_t *dma = (VOLATILE vmp_dma_descr_t *) __builtin_vmp_convert_wordaddresstobyteaddress(VMP_DMA_DESCR_ORG);

    dma->dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    dma->dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    dma->dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    dma->dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_READ | dmaChannelId);
}

/**
 * @brief Starts a write DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a write DMA transfer on the defined channel.
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
inline void vmp_dma_write(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    VOLATILE vmp_dma_descr_t *dma = (VOLATILE vmp_dma_descr_t *) __builtin_vmp_convert_wordaddresstobyteaddress(VMP_DMA_DESCR_ORG);

    dma->dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    dma->dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    dma->dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    dma->dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_WRITE | dmaChannelId);
}

// memset DMA functions with direct access to DMA descriptors
/**
 * @brief Starts a memset read DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a memset read DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param xferFlags_intWordaddr_lengthInt_length transfer flags (s3), internal word address (s2), length of segment internal 2D (s1) and length (s0)
 * @param reserved_strideInt_count3d_count reserved (s3), stride internal (s2), number of segments 3D (s1) and number of segments 2D (s0)
 * @param stride3d_stride stride 3D (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((always_inline))
/// @endcond
inline void vmp_dma_memset_read(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    VOLATILE vmp_dma_descr_t *dma = (VOLATILE vmp_dma_descr_t *) __builtin_vmp_convert_wordaddresstobyteaddress(VMP_DMA_DESCR_ORG);

    dma->dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    dma->dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    dma->dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    dma->dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_READ | VID_VMP_DMA_MEMSET | dmaChannelId);
}

/**
 * @brief Starts a memset write DMA transfer
 *
 * This function writes the input parameters to the corresponding DMA descriptors of the channel and
 * starts a memset write DMA transfer on the defined channel.
 *
 * @param extByteaddr external byte start address (s0 lower part, s1 higher part)
 * @param xferFlags_intWordaddr_lengthInt_length transfer flags (s3), internal word address (s2), length of segment internal 2D (s1) and length (s0)
 * @param reserved_strideInt_count3d_count reserved (s3), stride internal (s2), number of segments 3D (s1) and number of segments 2D (s0)
 * @param stride3d_stride stride 3D (s1) and stride (s0)
 * @param dmaChannelId DMA channel ID
 */
/// @cond
__attribute__((always_inline))
/// @endcond
inline void vmp_dma_memset_write(const uint2 extByteaddr,
                const ushort4 xferFlags_intWordaddr_lengthInt_length,
                const ushort4 reserved_strideInt_count3d_count,
                const uint2 stride3d_stride,
                const uint dmaChannelId)
{
    VOLATILE vmp_dma_descr_t *dma = (VOLATILE vmp_dma_descr_t *) __builtin_vmp_convert_wordaddresstobyteaddress(VMP_DMA_DESCR_ORG);

    dma->dma_descr[dmaChannelId].extByteaddr = extByteaddr;
    dma->dma_descr[dmaChannelId].xferFlags_intWordaddr_lengthInt_length = xferFlags_intWordaddr_lengthInt_length;
    dma->dma_descr[dmaChannelId].reserved_strideInt_count3d_count = reserved_strideInt_count3d_count;
    dma->dma_descr[dmaChannelId].stride3d_stride = stride3d_stride;

    __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_PRIO_STD | VID_VMP_DMA_WRITE | VID_VMP_DMA_MEMSET | dmaChannelId);
}

/// Wait for DMA transfer completion on specific channel
#define vmp_dma_wait_channel(__ch) (__builtin_vmp_wait((uint)1u << __ch))
/// Wait for DMA transfer completion on channel mask
#define vmp_dma_wait_mask(__ma) (__builtin_vmp_wait(__ma))

//lint -restore

#endif // __VID_VMP_DMA_H__
