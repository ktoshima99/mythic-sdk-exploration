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
 * FILENAME: vmp_cl-dma.h
 *
 * DESCRIPTION: interface to videantis v-MP 3.x DMA functions
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef __VMP_CL_DMA_H__
#define __VMP_CL_DMA_H__


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
#endif

//lint -save

//lint -e146 Error 146: Assuming a binary constant
//lintReason: vmpcc compiler supports binary constants

//lint -e9026 Note 9026: Function-like macro, 'vid_wait_group_*', defined [MISRA 2012 Directive 4.9, advisory]
//lintReason Lint exceptions introduced because called builtin requires constant fold to an integer constant expression

// avoid language incompatibility between OpenCL-C and C++ assumed by oclint
#ifdef __videantis_lint__
#define VOLATILE
#define PCAST long
#else
#define VOLATILE volatile
#define PCAST int
#endif

/// Number of DMA channels per each v-MP core
#define VID_VMP_NUM_DMA_CHANNELS 16u

#if !defined(VID_VMP_DMA_BASE)
/// Default internal base address for array of DMA channel descriptors
#define VID_VMP_DMA_BASE (0x600u - 64u)
#endif

/// v-MP bus interface unit internal control address
#define VID_VMP_BIU_BASE 0x3b1u

/// DMA transfer internal priority 0
#define VID_VMP_DMA_PRIO_INT0 (0b0000u << 8u)
/// DMA transfer internal priority 1
#define VID_VMP_DMA_PRIO_INT1 (0b0001u << 8u)
/// DMA transfer internal priority 2
#define VID_VMP_DMA_PRIO_INT2 (0b0010u << 8u)
/// DMA transfer internal priority 3
#define VID_VMP_DMA_PRIO_INT3 (0b0011u << 8u)
/// DMA transfer external priority 0
#define VID_VMP_DMA_PRIO_EXT0 (0b0000u << 8u)
/// DMA transfer external priority 1
#define VID_VMP_DMA_PRIO_EXT1 (0b0100u << 8u)
/// DMA transfer external priority 2
#define VID_VMP_DMA_PRIO_EXT2 (0b1000u << 8u)
/// DMA transfer external priority 3
#define VID_VMP_DMA_PRIO_EXT3 (0b1100u << 8u)
/// FIXME: DMA memset transfer
#define VID_VMP_DMA_MEMSET (1u << 5u)
/// DMA transfer direction read from external memory
#define VID_VMP_DMA_READ (1u << 4u)
/// DMA transfer direction write to external memory
#define VID_VMP_DMA_WRITE (0u << 4u)

/// @brief DMA channel descriptor struct data type
typedef struct {
  uint2 int_wordaddr__ext_byteaddr; ///< Internal word address (.s1) and external byte address (.s0)
  uint2 length__stride_count;       ///< Number of 64Bit words in each transfer line (.s1) and Stride in 64Bit words in external memory (.s0 >> 16) and Number of lines to transfer (.s0 && 0xffff)
} vmp_dma_t;

/// @brief All DMA channel descriptors placed at internal address VID_VMP_DMA_BASE
typedef struct {
  vmp_dma_t dma_descr[VID_VMP_NUM_DMA_CHANNELS]; ///< Array of 16 DMA channel descriptors
} vmp_dma_descr_t;

/// @brief union to support casting between different pointer types
/// required for DMA transfer function
//lint -e9018 -e9087
//lintNote 9018: declaration of symbol 'unknown-name' with union based type 'const vmp_caster_t &' [MISRA 2012 Rule 19.2, advisory]
//lintReason: union type required to allow valid pointer cast for DMA transfer functions
typedef union {
    __local void* lvoid;
    __local ulong1* lulong1;
    __local uint2* luint2;
    __local ushort4* lushort4;
    __local uchar8* luchar8;
    __local long1* llong1;
    __local int2* lint2;
    __local short4* lshort4;
    __local char8* lchar8;
    void* pvoid;
    ulong1* pulong1;
    uint2* puint2;
    ushort4* pushort4;
    uchar8* puchar8;
    long1* plong1;
    int2* pint2;
    short4* pshort4;
    char8* pchar8;
} vmp_caster_t;

/** @brief Start 2D DMA transfer to read 2D data from */
/** external memory and transfer to internal memory */
/** @param dst Pointer to local data memory */
/** @param src External byte address in global memory */
/** @param num_gentypes_width Number of 64Bit words in each transfer line */
/** @param num_lines_height Number of lines to transfer */
/** @param num_gentypes_stride Stride in 64Bit words in external memory */
/** @param dma_channel_id DMA channel ID */
__attribute__((overloadable, always_inline)) inline void
vmp_strided_read_2D(__byteaddress __local void *const dst, const uint src,
                    const uint num_gentypes_width, const uint num_lines_height,
                    const uint num_gentypes_stride, const uint dma_channel_id);
__attribute__((overloadable, always_inline)) inline void
vmp_strided_read_2D(__byteaddress __local void *const dst, const uint src,
                    const uint num_gentypes_width, const uint num_lines_height,
                    const uint num_gentypes_stride, const uint dma_channel_id) {
  VOLATILE vmp_dma_descr_t * const dma = (VOLATILE vmp_dma_descr_t * const)
      __builtin_vmp_convert_wordaddresstobyteaddress(VID_VMP_DMA_BASE);

  // int_wordaddr__ext_byteaddr
  dma->dma_descr[dma_channel_id].int_wordaddr__ext_byteaddr = set_uint2(
      src, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)dst));

  // length__stride_count
  dma->dma_descr[dma_channel_id].length__stride_count = set_uint2(
      (num_gentypes_stride << 16u) | num_lines_height, num_gentypes_width);

  __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_READ | dma_channel_id);
}

/** @brief Start 2D DMA transfer to read data from */
/** internal memory and transfer 2D to external memory */
/** @param dst External byte address in global memory */
/** @param src Pointer to local data memory */
/** @param num_gentypes_width Number of 64Bit words in each transfer line */
/** @param num_lines_height Number of lines to transfer */
/** @param num_gentypes_stride Stride in 64Bit words in external memory */
/** @param dma_channel_id DMA channel ID */
__attribute__((overloadable, always_inline)) inline void
vmp_strided_write_2D(const uint dst, __byteaddress __local void *const src,
                     const uint num_gentypes_width, const uint num_lines_height,
                     const uint num_gentypes_stride, const uint dma_channel_id);
__attribute__((overloadable, always_inline)) inline void
vmp_strided_write_2D(const uint dst, __byteaddress __local void *const src,
                     const uint num_gentypes_width, const uint num_lines_height,
                     const uint num_gentypes_stride,
                     const uint dma_channel_id) {
  VOLATILE vmp_dma_descr_t *const dma = (VOLATILE vmp_dma_descr_t *const)
      __builtin_vmp_convert_wordaddresstobyteaddress(VID_VMP_DMA_BASE);

  // int_wordaddr__ext_byteaddr
  dma->dma_descr[dma_channel_id].int_wordaddr__ext_byteaddr = set_uint2(
      dst, (uint)__builtin_vmp_convert_byteaddresstowordaddress((PCAST)src));

  // length__stride_count
  dma->dma_descr[dma_channel_id].length__stride_count = set_uint2(
      (num_gentypes_stride << 16u) | num_lines_height, num_gentypes_width);

  __builtin_vmp_write_BIU_DMA_CTRL(VID_VMP_DMA_WRITE | dma_channel_id);
}

#define DMA_TRANSFERS(_LOCAL, _TYPE, _CASTER)                                  \
  /** @brief Start linear DMA transfer to read data from */                    \
  /** external memory and transfer to internal memory */                       \
  /** @param dst Pointer to local data memory */                               \
  /** @param src External byte address in global memory */                     \
  /** @param num_gentypes Number of 64Bit words to transfer */                 \
  /** @param dma_channel_id DMA channel ID */                                  \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_copy(__byteaddress _LOCAL _TYPE *const dst,             \
                            const uint src, const uint num_gentypes,           \
                            const uint dma_channel_id);                        \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_copy(__byteaddress _LOCAL _TYPE *const dst,             \
                            const uint src, const uint num_gentypes,           \
                            const uint dma_channel_id) {                       \
    vmp_caster_t c;                                                            \
    c._CASTER = dst;                                                           \
    vmp_strided_read_2D(c.lvoid, src, num_gentypes, 1u, 0u, dma_channel_id);   \
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
      __byteaddress _LOCAL _TYPE *const dst, const uint src,                   \
      const uint num_gentypes_width, const uint num_lines_height,              \
      const uint num_gentypes_stride, const uint dma_channel_id);              \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_2D_copy(                                                \
      __byteaddress _LOCAL _TYPE *const dst, const uint src,                   \
      const uint num_gentypes_width, const uint num_lines_height,              \
      const uint num_gentypes_stride, const uint dma_channel_id) {             \
    vmp_caster_t c;                                                            \
    c._CASTER = dst;                                                           \
    vmp_strided_read_2D(c.lvoid, src, num_gentypes_width, num_lines_height,    \
                        num_gentypes_stride, dma_channel_id);                  \
  }                                                                            \
                                                                               \
  /** @brief Start linear DMA transfer to write data from */                   \
  /** internal memory and transfer to external memory */                       \
  /** @param dst External byte address in global memory */                     \
  /** @param src Pointer to local data memory */                               \
  /** @param num_gentypes Number of 64Bit words to transfer */                 \
  /** @param dma_channel_id DMA channel ID */                                  \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_copy(                                                   \
      const uint dst, __byteaddress _LOCAL _TYPE *const src,                   \
      const uint num_gentypes, const uint dma_channel_id);                     \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_copy(                                                   \
      const uint dst, __byteaddress _LOCAL _TYPE *const src,                   \
      const uint num_gentypes, const uint dma_channel_id) {                    \
    vmp_caster_t c;                                                            \
    c._CASTER = src;                                                           \
    vmp_strided_write_2D(dst, c.lvoid, num_gentypes, 1u, 0u, dma_channel_id);  \
  }                                                                            \
                                                                               \
  /** @brief Start 2D DMA transfer to read data from */                        \
  /** internal memory and transfer 2D to external memory */                    \
  /** @param dst External byte address in global memory */                     \
  /** @param src Pointer to local data memory */                               \
  /** @param num_gentypes_width Number of 64Bit words in each transfer line */ \
  /** @param num_lines_height Number of lines to transfer */                   \
  /** @param num_gentypes_stride Stride in 64Bit words in external memory */   \
  /** @param dma_channel_id DMA channel ID */                                  \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_2D_copy(                                                \
      const uint dst, __byteaddress _LOCAL _TYPE *const src,                   \
      const uint num_gentypes_width, const uint num_lines_height,              \
      const uint num_gentypes_stride, const uint dma_channel_id);              \
  __attribute__((overloadable, always_inline)) inline void                     \
  vid_async_work_group_2D_copy(                                                \
      const uint dst, __byteaddress _LOCAL _TYPE *const src,                   \
      const uint num_gentypes_width, const uint num_lines_height,              \
      const uint num_gentypes_stride, const uint dma_channel_id) {             \
    vmp_caster_t c;                                                            \
    c._CASTER = src;                                                           \
    vmp_strided_write_2D(dst, c.lvoid, num_gentypes_width, num_lines_height,   \
                         num_gentypes_stride, dma_channel_id);                 \
  }

/*
 * DMA transfer functions for all supported data types
 */
#ifdef __VMP__
DMA_TRANSFERS(, void, pvoid)
DMA_TRANSFERS(, ulong1, pulong1)
DMA_TRANSFERS(, uint2, puint2)
DMA_TRANSFERS(, ushort4, pushort4)
DMA_TRANSFERS(, uchar8, puchar8)
DMA_TRANSFERS(, long1, plong1)
DMA_TRANSFERS(, int2, pint2)
DMA_TRANSFERS(, short4, pshort4)
DMA_TRANSFERS(, char8, pchar8)
#endif
DMA_TRANSFERS(__local, void, lvoid)
DMA_TRANSFERS(__local, ulong1, lulong1)
DMA_TRANSFERS(__local, uint2, luint2)
DMA_TRANSFERS(__local, ushort4, lushort4)
DMA_TRANSFERS(__local, uchar8, luchar8)
DMA_TRANSFERS(__local, long1, llong1)
DMA_TRANSFERS(__local, int2, lint2)
DMA_TRANSFERS(__local, short4, lshort4)
DMA_TRANSFERS(__local, char8, lchar8)

/// Wait for DMA transfer completion on specific channel
#define vid_wait_group_channel(__chx) (__builtin_vmp_wait((uint)1u << __chx))

/// Wait for DMA transfer completion on channel mask
#define vid_wait_group_mask(__chx) (__builtin_vmp_wait(__chx))

//lint -restore

#endif // __VMP_CL_DMA_H__
