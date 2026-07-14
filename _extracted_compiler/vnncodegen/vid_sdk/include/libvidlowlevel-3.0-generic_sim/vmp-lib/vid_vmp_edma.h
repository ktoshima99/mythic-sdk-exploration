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
 * FILENAME: vid_vmp_edma.h
 *
 * DESCRIPTION: videantis-C v-MP 4.x EDMA definitions and generic functions
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @file
 * @brief videantis-C v-MP 4.x EDMA definitions and generic functions
 *
 * This file provides definitions and always inline generic functions for EDMA transfer
 *
 */

#ifndef __VID_VMP_EDMA__
#define __VID_VMP_EDMA__

// vector data types
#include "vmp_cl/vmp_cl-types.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cl/vid_videantis-C.hpp"
#endif

// videantis-C v-MP 4.x DMA definitions and generic functions
#include "vmp-lib/vid_vmp_dma.h"

// hardware-specific constants
#include "target_consts.h"

// definitions for little endian DMA transfer functions
#include "vid_vmp_endianness_dma_func.h"

/// number of EDMA channels
#define VID_VMP_NUM_EDMA_CHANNELS 8u
// EDMA channel descriptors base address
// if not defined place before DMA descriptors and memory for legacy DMA transfer functions high part
#ifndef VMP_EDMA_DESCR_ORG
/// default base address of EDMA descriptors
#define VMP_EDMA_DESCR_ORG (VMP_DMA_DESCR_ORG - 7u)
#endif

/// DMA channel used for transfers related to EDMA
#define VID_VMP_EDMA_DMA_CHANNEL 15u

/// Standard flag
#define VID_VMP_EDMA_STD       (0u)
/// Enable 24-to-32-bit expansion flag: insert zero byte after three source bytes
#define VID_VMP_EDMA_EXP_24_32 (1u)

/// EDMA register channel size in bytes
#define EDMA_REG_CHANNEL_SIZE   16u
/// EDMA register status offset in bytes
#define EDMA_REG_STATUS_OFFSET  8u

/**
 * @struct vid_edma_descr_t
 * @brief Structure contains EDMA channel descriptor
 *
 * EDMA descriptor layout
 *
 * | Name       | Byte Offs. | Bits | Description                                                              |
 * |------------|------------|------|--------------------------------------------------------------------------|
 * | TGT_ADDR   | 0x00..0x07 | 64   | Target start address (byte address)                                      |
 * | SRC_ADDR   | 0x08..0x0F | 64   | Source start address (byte address)                                      |
 * | LENGTH     | 0x10..0x13 | 32   | Length in bytes; for EXP_24_32 mode: target bytes, must be multiple of 4 |
 * | COUNT      | 0x14..0x15 | 16   | Number of 2D segments (0 or 1 => 1D)                                     |
 * | FLAGS      | 0x16..0x17 | 16   | Transfer flags                                                           |
 * | TGT_STRIDE | 0x18..0x1B | 32   | Target 2D segment stride (bytes; not used for 1D)                        |
 * | SRC_STRIDE | 0x1C..0x1F | 32   | Source 2D segment stride (bytes; not used for 1D)
 *
 * Details of FLAGS field in EDMA descriptor
 *
 * | Bit pos. | Bits | Name       | Description                                                              |
 * |----------|------|------------|--------------------------------------------------------------------------|
 * | 0        | 1    | EXP_24_32  | Enable 24-to-32-bit expansion: insert zero byte after three source bytes |
 * | 15..1    | -    | (reserved) | Reserved for future use, must be 0                                       |
 *
 * The subwords are reordered in the structure to match big endian architecture of v-MP,
 * when transferred to external memory with a LE64 DMA transfer.
 *
 * @var uint2 vid_edma_descr_t::tgtAddr
 * Contains the external byte target address (s0 lower part of the address, s1 higher part of the address)
 * @var uint2 vid_edma_descr_t::srcAddr
 * Contains the external byte source address (s0 lower part of the address, s1 higher part of the address)
 * @var ushort4 vid_edma_descr_t::t.flags_count_lengthHi_lengthLo
 * Contains the flags (s3), number of 2D segments (s2) and length in bytes (s1)+(s0)
 * @var uint2 vid_edma_descr_t::t.count_length
 * Contains the number of 2D segments (s1) and length in bytes (s0) (simplified access with flags = 0)
 * @var uint2 vid_edma_descr_t::srcStride_tgtStride
 * Contains the distance (number of bytes) between the start address of two consecutive source segments (s1) and
 * distance (number of bytes) between the start address of two consecutive target segments (s0)
 */
typedef struct {
    uint2 tgtAddr;
    uint2 srcAddr;
    union {
        ushort4 flags_count_lengthHi_lengthLo;
        uint2 count_length;
    } t;
    uint2 srcStride_tgtStride;
} vid_edma_descr_t;

/*
 * EDMA generic functions
 */

/**
 * @brief Starts a EDMA transfer
 *
 * This function writes the address of EDMA descriptor to EDMA descriptor register
 * to start a transfer on the defined EDMA channel.
 *
 * @param extDescrAddr external descriptor byte address (s0 lower part, s1 higher part))
 * @param edmaChannelId EDMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_edma(const uint2 extDescrAddr,
                const uint edmaChannelId)
{
    VOLATILE uint2 *extDescrAddrPtr = (VOLATILE uint2 *) __builtin_vmp_convert_wordaddresstobyteaddress(VMP_EDMA_DESCR_ORG + SIZEOF_IN_WORDS(vid_edma_descr_t));

    // save external descriptor address into memory
    *extDescrAddrPtr = extDescrAddr;

    // write external EDMA descriptor address into EDMA register
    uint2 edmaReg;
    edmaReg.s1 = (VMP_CTRL_BASE + EDMA_REGWIN_BASE) >> 32u;
    edmaReg.s0 = (VMP_CTRL_BASE + EDMA_REGWIN_BASE + (edmaChannelId * EDMA_REG_CHANNEL_SIZE)) & 0xFFFFFFFFu;

    const uint2 edmaRegintWordaddr_length = set_uint2(1u, (uint)__builtin_vmp_convert_byteaddresstowordaddress((int)extDescrAddrPtr));

    vmp_dma_write_LE64(edmaReg, as_ushort4(edmaRegintWordaddr_length), set_ushort4(0u), set_uint2(0u), VID_VMP_EDMA_DMA_CHANNEL);

    vmp_dma_wait_channel(VID_VMP_EDMA_DMA_CHANNEL);
}

/**
 * @brief Starts a EDMA transfer
 *
 * This function writes the input parameter to the corresponding EDMA descriptor
 * in external memory and writes the address of EDMA descriptor to EDMA descriptor
 * register to start a transfer on the defined EDMA channel.
 *
 * @param tgtAddr external target byte start address (s0 lower part, s1 higher part)
 * @param srcAddr external source byte start address (s0 lower part, s1 higher part)
 * @param flags_count_lengthHi_lengthLo transfer flags (s3), number of segments 2D (s2) and length (s0 lower part, s1 higher part)
 * @param srcStride_tgtStride source stride (s1) and target stride (s0)
 * @param extDescrAddr external descriptor byte address (s0 lower part, s1 higher part))
 * @param edmaChannelId EDMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_edma(const uint2 tgtAddr,
                const uint2 srcAddr,
                const ushort4 flags_count_lengthHi_lengthLo,
                const uint2 srcStride_tgtStride,
                const uint2 extDescrAddr,
                const uint edmaChannelId)
{
    VOLATILE vid_edma_descr_t *edma = (VOLATILE vid_edma_descr_t *) __builtin_vmp_convert_wordaddresstobyteaddress(VMP_EDMA_DESCR_ORG);

    // write descriptor data into memory
    edma->tgtAddr = tgtAddr;
    edma->srcAddr = srcAddr;
    edma->t.flags_count_lengthHi_lengthLo = flags_count_lengthHi_lengthLo;
    edma->srcStride_tgtStride = srcStride_tgtStride;

    // setup DMA transfer to transfer EDMA descriptor into external memory
    const uint2 intWordaddr_length = set_uint2(SIZEOF_IN_WORDS(vid_edma_descr_t), VMP_EDMA_DESCR_ORG);

    // write EDMA descriptor to external EDMA descriptor
    vmp_dma_write_LE64(extDescrAddr, as_ushort4(intWordaddr_length), set_ushort4(0u), set_uint2(0u), VID_VMP_EDMA_DMA_CHANNEL);

    vmp_dma_wait_channel(VID_VMP_EDMA_DMA_CHANNEL);

    // setup DMA transfer to read external EDMA descriptor back to ensure the data is written correctly
    const uint2 writeBackintWordaddr_length = set_uint2(1u, VMP_EDMA_DESCR_ORG + SIZEOF_IN_WORDS(vid_edma_descr_t) + 1u);

    // read one word back from external EDMA descriptor
    vmp_dma_read_LE64(extDescrAddr, as_ushort4(writeBackintWordaddr_length), set_ushort4(0u), set_uint2(0u), VID_VMP_EDMA_DMA_CHANNEL);

    vmp_dma_wait_channel(VID_VMP_EDMA_DMA_CHANNEL);

    // write address of external EDMA descriptor to EDMA register
    vmp_edma(extDescrAddr, edmaChannelId);
}

/**
 * @brief Starts a EDMA transfer with flags = 0
 *
 * This function writes the simplified input parameter to the corresponding EDMA descriptor
 * in external memory and writes the address of EDMA descriptor to EDMA descriptor
 * register to start a transfer on the defined EDMA channel.
 *
 * @param tgtAddr external target byte start address (s0 lower part, s1 higher part)
 * @param srcAddr external source byte start address (s0 lower part, s1 higher part)
 * @param count_length number of segments 2D (s1) and length (s0)
 * @param srcStride_tgtStride source stride (s1) and target stride (s0)
 * @param extDescrAddr external descriptor byte address (s0 lower part, s1 higher part))
 * @param edmaChannelId EDMA channel ID
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vmp_edma(const uint2 tgtAddr,
                const uint2 srcAddr,
                const uint2 count_length,
                const uint2 srcStride_tgtStride,
                const uint2 extDescrAddr,
                const uint edmaChannelId)
{
    VOLATILE vid_edma_descr_t *edma = (VOLATILE vid_edma_descr_t *) __builtin_vmp_convert_wordaddresstobyteaddress(VMP_EDMA_DESCR_ORG);

    // write descriptor data into memory
    edma->tgtAddr = tgtAddr;
    edma->srcAddr = srcAddr;
    edma->t.count_length = count_length;
    edma->srcStride_tgtStride = srcStride_tgtStride;

    // setup DMA transfer to transfer EDMA descriptor into external memory
    const uint2 intWordaddr_length = set_uint2(SIZEOF_IN_WORDS(vid_edma_descr_t), VMP_EDMA_DESCR_ORG);

    // write EDMA descriptor to external EDMA descriptor
    vmp_dma_write_LE64(extDescrAddr, as_ushort4(intWordaddr_length), set_ushort4(0u), set_uint2(0u), VID_VMP_EDMA_DMA_CHANNEL);

    vmp_dma_wait_channel(VID_VMP_EDMA_DMA_CHANNEL);

    // setup DMA transfer to read external EDMA descriptor back to ensure the data is written correctly
    const uint2 writeBackintWordaddr_length = set_uint2(1u, VMP_EDMA_DESCR_ORG + SIZEOF_IN_WORDS(vid_edma_descr_t) + 1u);

    // read one word back from external EDMA descriptor
    vmp_dma_read_LE64(extDescrAddr, as_ushort4(writeBackintWordaddr_length), set_ushort4(0u), set_uint2(0u), VID_VMP_EDMA_DMA_CHANNEL);

    vmp_dma_wait_channel(VID_VMP_EDMA_DMA_CHANNEL);

    // write address of external EDMA descriptor to EDMA register
    vmp_edma(extDescrAddr, edmaChannelId);
}

/**
 * @brief Checks a EDMA channel status
 *
 * This function returns the EDMA channel status register.
 *
 * @param edmaChannelId EDMA channel ID
 * @return Status of EDMA channel
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline uint vmp_edma_check(const uint edmaChannelId)
{
    VOLATILE uint2 *edmaStatusPtr = (VOLATILE uint2 *) __builtin_vmp_convert_wordaddresstobyteaddress(VMP_EDMA_DESCR_ORG + SIZEOF_IN_WORDS(vid_edma_descr_t) + 1u);

    // setup DMA transfer to read status registers of defined EDMA channel
    uint2 edmaRegStatus;
    edmaRegStatus.s1 = (VMP_CTRL_BASE + EDMA_REGWIN_BASE) >> 32u;
    edmaRegStatus.s0 = (VMP_CTRL_BASE + EDMA_REGWIN_BASE + (edmaChannelId * EDMA_REG_CHANNEL_SIZE) + EDMA_REG_STATUS_OFFSET) & 0xFFFFFFFFu;

    const uint2 edmaStatusintWordaddr_length = set_uint2(1u, (uint)__builtin_vmp_convert_byteaddresstowordaddress((int)edmaStatusPtr));

    // read status register of EDMA channel via DMA
    vmp_dma_read_LE64(edmaRegStatus, as_ushort4(edmaStatusintWordaddr_length), set_ushort4(0u), set_uint2(0u), VID_VMP_EDMA_DMA_CHANNEL);

    vmp_dma_wait_channel(VID_VMP_EDMA_DMA_CHANNEL);

    return edmaStatusPtr->s0;
}

#endif // __VID_VMP_EDMA__
