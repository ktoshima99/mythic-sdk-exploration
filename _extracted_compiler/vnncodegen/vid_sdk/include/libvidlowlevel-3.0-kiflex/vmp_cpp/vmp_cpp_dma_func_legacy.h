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
 * FILENAME: vmp_cpp_dma_func_legacy.h
 *
 * DESCRIPTION: videantis-C v-MP 4.x DMA legacy functions
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @file
 * @brief videantis-C v-MP 4.x DMA legacy functions
 *
 * This file provides always inline legacy functions for DMA transfers
 *
 */

#ifndef VMP_CPP_DMA_FUNC_LEGACY_H
#define VMP_CPP_DMA_FUNC_LEGACY_H

// vector data types
#include "vmp_cpp-types.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cpp_videantis-C.h"
#endif

// core DMA transfer functions
#include "vmp_cpp_dma.h"

//lint -save

//lint -e9026 Note 9026: Function-like macro, 'vid_wait_group_*', defined [MISRA 2012 Directive 4.9, advisory]
//lintReason Lint exceptions introduced because called builtin requires constant fold to an integer constant expression

// avoid language incompatibility between OpenCL-C and C++ pointer sizes assumed by oclint
#ifdef __videantis_lint__
#define PCAST long
#else
#define PCAST int
#endif

/// @brief legacy DMA transfer high part of 64bit external addresses
extern volatile uint2 vid_dma_legacy_high;

#define DMA_TRANSFERS(_TYPE, _CASTER)                                          \
  /** @brief Start linear DMA transfer to read data from */                    \
  /** external memory and transfer to internal memory */                       \
  /** @param dst Pointer to local data memory */                               \
  /** @param src External byte address in global memory */                     \
  /** @param num_gentypes Number of 64Bit words to transfer */                 \
  /** @param dma_channel_id DMA channel ID */                                  \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_copy(__byteaddress _TYPE *const dst, const uint src,    \
                            const uint num_gentypes,                           \
                            const uint dma_channel_id);                        \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_copy(__byteaddress _TYPE *const dst, const uint src,    \
                            const uint num_gentypes,                           \
                            const uint dma_channel_id) {                       \
    vmp_dma_read(set_uint2(src, vid_dma_legacy_high.s0),                       \
                 as_ushort4(set_uint2(                                         \
                     num_gentypes,                                             \
                     (uint)__builtin_vmp_convert_byteaddresstowordaddress(     \
                         (PCAST)dst))),                                        \
                 set_ushort4(0u), set_uint2(0u), dma_channel_id);              \
  }                                                                            \
                                                                               \
  /** @brief Start 2D DMA transfer to read 2D data from */                     \
  /** external memory and transfer to internal memory */                       \
  /** @param dst Pointer to local data memory */                               \
  /** @param src External byte address in global memory */                     \
  /** @param num_gentypes_width Number of 64Bit words in each transfer line */ \
  /** @param num_lines_height Number of lines to transfer */                   \
  /** @param num_gentypes_stride Stride in 64Bit words in external memory */   \
  /** @param dma_channel_id DMA channel ID */                                  \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_2D_copy(                                                \
      __byteaddress _TYPE *const dst, const uint src,                          \
      const uint num_gentypes_width, const uint num_lines_height,              \
      const uint num_gentypes_stride, const uint dma_channel_id);              \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_2D_copy(                                                \
      __byteaddress _TYPE *const dst, const uint src,                          \
      const uint num_gentypes_width, const uint num_lines_height,              \
      const uint num_gentypes_stride, const uint dma_channel_id) {             \
    vmp_dma_read(set_uint2(src, vid_dma_legacy_high.s0),                       \
                 as_ushort4(set_uint2(                                         \
                     num_gentypes_width,                                       \
                     (uint)__builtin_vmp_convert_byteaddresstowordaddress(     \
                         (PCAST)dst))),                                        \
                 as_ushort4(set_uint2(num_lines_height, 0u)),                  \
                 set_uint2(num_gentypes_stride, 0u), dma_channel_id);          \
  }                                                                            \
                                                                               \
  /** @brief Start linear DMA transfer to write data from */                   \
  /** internal memory and transfer to external memory */                       \
  /** @param dst External byte address in global memory */                     \
  /** @param src Pointer to local data memory */                               \
  /** @param num_gentypes Number of 64Bit words to transfer */                 \
  /** @param dma_channel_id DMA channel ID */                                  \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_copy(const uint dst, __byteaddress _TYPE *const src,    \
                            const uint num_gentypes,                           \
                            const uint dma_channel_id);                        \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_copy(const uint dst, __byteaddress _TYPE *const src,    \
                            const uint num_gentypes,                           \
                            const uint dma_channel_id) {                       \
    vmp_dma_write(set_uint2(dst, vid_dma_legacy_high.s0),                      \
                  as_ushort4(set_uint2(                                        \
                      num_gentypes,                                            \
                      (uint)__builtin_vmp_convert_byteaddresstowordaddress(    \
                          (PCAST)src))),                                       \
                  set_ushort4(0u), set_uint2(0u), dma_channel_id);             \
  }                                                                            \
                                                                               \
  /** @brief Start 2D DMA transfer to write data from */                       \
  /** internal memory and transfer 2D to external memory */                    \
  /** @param dst External byte address in global memory */                     \
  /** @param src Pointer to local data memory */                               \
  /** @param num_gentypes_width Number of 64Bit words in each transfer line */ \
  /** @param num_lines_height Number of lines to transfer */                   \
  /** @param num_gentypes_stride Stride in 64Bit words in external memory */   \
  /** @param dma_channel_id DMA channel ID */                                  \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_2D_copy(                                                \
      const uint dst, __byteaddress _TYPE *const src,                          \
      const uint num_gentypes_width, const uint num_lines_height,              \
      const uint num_gentypes_stride, const uint dma_channel_id);              \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_2D_copy(                                                \
      const uint dst, __byteaddress _TYPE *const src,                          \
      const uint num_gentypes_width, const uint num_lines_height,              \
      const uint num_gentypes_stride, const uint dma_channel_id) {             \
    vmp_dma_write(set_uint2(dst, vid_dma_legacy_high.s0),                      \
                  as_ushort4(set_uint2(                                        \
                      num_gentypes_width,                                      \
                      (uint)__builtin_vmp_convert_byteaddresstowordaddress(    \
                          (PCAST)src))),                                       \
                  as_ushort4(set_uint2(num_lines_height, 0u)),                 \
                  set_uint2(num_gentypes_stride, 0u), dma_channel_id);         \
  }

/*
 * DMA transfer functions for all supported data types
 */
DMA_TRANSFERS(void, pvoid)
DMA_TRANSFERS(ulong1, pulong1)
DMA_TRANSFERS(uint2, puint2)
DMA_TRANSFERS(ushort4, pushort4)
DMA_TRANSFERS(uchar8, puchar8)
DMA_TRANSFERS(long1, plong1)
DMA_TRANSFERS(int2, pint2)
DMA_TRANSFERS(short4, pshort4)
DMA_TRANSFERS(char8, pchar8)

/**
 * @brief Set external byte address high part for legacy dma transfer functions
 *
 * @param extByteaddrHigh external byte address high part of 64-bit address
 */
/// @cond
__attribute__((overloadable, always_inline))
/// @endcond
inline void vid_dma_legacy_addr_high(const uint extByteaddrHigh)
{
  vid_dma_legacy_high = set_uint2(extByteaddrHigh);
}

// DMA wait legacy functions
/// Wait for DMA transfer completion on specific channel
#define vid_wait_group_channel(__ch) (__builtin_vmp_wait((uint)1 << __ch))
/// Wait for DMA transfer completion on channel mask
#define vid_wait_group_mask(__ma) (__builtin_vmp_wait(__ma))

//lint -restore

#endif // VMP_CPP_DMA_FUNC_LEGACY_H
