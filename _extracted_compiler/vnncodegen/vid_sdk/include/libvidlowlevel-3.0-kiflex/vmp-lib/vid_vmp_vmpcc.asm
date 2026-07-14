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
 * FILENAME: vid_vmp_vmpcc.asm
 *
 * DESCRIPTION: videantis v-MP compiler support functions and data sections
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP compiler support functions and data sections
 *
 * @details
 * This file implements memory functions memcmp(), memcpy(), memset() with
 * prototypes for these functions available in vmp_cl-string.h.
 * This file implements exit(int exitcode).
 * This file provides vid_vmp_permutation_masks, mulhu, mulhs used by the compiler.
 * This file implements __builtin_vmp_clz_8, __builtin_vmp_clz_16, __builtin_vmp_clz_32.
 * This file implements __builtin_vmp_ctz_8, __builtin_vmp_ctz_16, __builtin_vmp_ctz_32.
 * This file implements __builtin_vmp_popcount_8, __builtin_vmp_popcount_16, __builtin_vmp_popcount_32.
 *
 * @file vid_vmp_vmpcc.asm
 */

/// @cond DOXYGEN_IGNORE_ASM


/* ========================================================================== *
 *         Exit section                                                       *
 * ========================================================================== */
.csection vid_vmp_exit
.export _exit

/**
 * @brief Exit and run to BREAK instruction
 *
 * This functions exits the current application. The exitCode is written to GPDATA0.
 * The core executes the BREAK instruction which causes the v-MP processor to halt.
 */
//void exit(int exitCode);
_exit:
    MV    GPDATA0, sparam0
    break
.endsection


/* ========================================================================== *
 *         constant data section                                              *
 * ========================================================================== */
.dsection vid_vmp_permutation_masks, dmem2
permmask_le_i8:
    .alloc [8] = { 0x76543218, 0x76543280, 0x76543810, 0x76548210,
            0x76583210, 0x76843210, 0x78543210, 0x86543210 }
.endsection

.dsection vid_vmp_permutation_masks, dmem2
permmask_le_i16:
    .alloc [4] = { 0x03020108, 0x03020800, 0x03080100, 0x08020100 }
.endsection

.dsection vid_vmp_permutation_masks, dmem2
permmask_le_i32:
    .alloc [2] = { 0x00010008, 0x00080000 }
.endsection


/**
 * @brief Upper 32Bits result of 32Bit times 32Bit unsigned integer multiplication
 *
 * This macro returns the upper 32Bits of the multiplication ((uint_64) _op1 * (uint_64) _op1) >> 32u
 * This macro is used by the compiler vmpcc to optimize uint modulo operations where the
 * second operator is a constant, e.g. temp %= 3u.
 */
.macro mulhu(_dst, _op1, _op2)
    MV mvtmp0, _op1
    MV mvtmp1, _op2
    V_MACPL0_U32 mvtmp0, mvtmp0, mvtmp1
    V_SRI_U64 mvtmp1, mvtmp1, #32
    MV _dst, mvtmp1
.endmacro

/**
 * @brief Upper 32Bits result of 32Bit times 32Bit signed integer multiplication
 *
 * This macro returns the upper 32Bits of the multiplication ((int_64) _op1 * (int_64) _op1) >> 32
 * This macro is used by the compiler vmpcc to optimize int modulo operations where the
 * second operator is a constant, e.g. temp %= 3.
 */
.macro mulhs(_dst, _op1, _op2)
    MV mvtmp0, _op1
    MV mvtmp1, _op2
    V_MACPL0_32 mvtmp0, mvtmp0, mvtmp1
    V_SRI_64 mvtmp1, mvtmp1, #32
    MV _dst, mvtmp1
.endmacro

.equ vn     = mvtmp0
.equ vshift = mvtmp1

/**
 * @brief Count leading zeros on data type uchar8
 *
 * Macro to count leading zeros on data type uchar8 called by __builtin_vmp_clz_8(uchar8 x)
*/
.macro vmp_clz_8(_dst, _op)
// use count matching bits instruction for v-MP core version 4.1 or newer
.if defined(__core_version__) && {__core_version__ >= 401}
    V_CMBI_8    _dst, _op, #0
.else
    V_MV_8     vshift, _op
    V_MVI_8    vn, #1

    MVI VCONDSEL, #COND_Z
    V_SRICS_U8     ZERO, vshift, #4
    V_ADDICR_U8    vn, vn, #4
    V_SLICR_U8     vshift, vshift, #4

    V_SRICS_U8     ZERO, vshift, #6
    V_ADDICR_U8    vn, vn, #2
    V_SLICR_U8     vshift, vshift, #2

    V_SUBICS_U8    ZERO, _op, #0

    V_SRI_U8       vshift, vshift, #7
    V_SUB_U8       _dst, vn, vshift

    V_MVICR_8      _dst, #8
.endif
.endmacro

/**
 * @brief Count leading zeros on data type ushort4
 *
 * Macro to count leading zeros on data type ushort4 called by __builtin_vmp_clz_16(ushort4 x)
*/
.macro vmp_clz_16(_dst, _op)
// use count matching bits instruction for v-MP core version 4.1 or newer
.if defined(__core_version__) && {__core_version__ >= 401}
    V_CMBI_16   _dst, _op, #0
.else
    V_MV_16     vshift, _op
    V_MVI_16    vn, #1

    MVI VCONDSEL, #COND_Z
    V_SRICS_U16     ZERO, vshift, #8
    V_ADDICR_U16    vn, vn, #8
    V_SLICR_U16     vshift, vshift, #8

    V_SRICS_U16     ZERO, vshift, #12
    V_ADDICR_U16    vn, vn, #4
    V_SLICR_U16     vshift, vshift, #4

    V_SRICS_U16     ZERO, vshift, #14
    V_ADDICR_U16    vn, vn, #2
    V_SLICR_U16     vshift, vshift, #2

    V_SUBICS_U16    ZERO, _op, #0

    V_SRI_U16       vshift, vshift, #15
    V_SUB_U16       _dst, vn, vshift

    V_MVICR         _dst, #16
.endif
.endmacro

/**
 * @brief Count leading zeros on data type uint2
 *
 * Macro to count leading zeros on data type uint2 called by __builtin_vmp_clz_32(uint2 x)
*/
.macro vmp_clz_32(_dst, _op)
// use count matching bits instruction for v-MP core version 4.1 or newer
.if defined(__core_version__) && {__core_version__ >= 401}
    V_CMBI_32   _dst, _op, #0
.else
    V_MV_32     vshift, _op
    V_MVI_32    vn, #1

    MVI VCONDSEL, #COND_Z
    V_SRICS_U32     ZERO, vshift, #16
    V_ADDICR_U32    vn, vn, #16
    V_SLICR_U32     vshift, vshift, #16

    V_SRICS_U32     ZERO, vshift, #24
    V_ADDICR_U32    vn, vn, #8
    V_SLICR_U32     vshift, vshift, #8

    V_SRICS_U32     ZERO, vshift, #28
    V_ADDICR_U32    vn, vn, #4
    V_SLICR_U32     vshift, vshift, #4

    V_SRICS_U32     ZERO, vshift, #30
    V_ADDICR_U32    vn, vn, #2
    V_SLICR_U32     vshift, vshift, #2

    V_SUBICS_U32    ZERO, _op, #0

    V_SRI_U32       vshift, vshift, #31
    V_SUB_U32       _dst, vn, vshift

    V_MVICR_32      _dst, #32
.endif
.endmacro

/**
 * @brief Count trailing zeros on data type uchar8
 *
 * Macro to count trailing zeros on data type uchar8 called by __builtin_vmp_ctz_8(uchar8 x)
*/
.macro vmp_ctz_8(_dst, _op)
    V_MV_8 vshift, _op
    V_MVI_8 vn, #0

    MVI VCONDSEL, #COND_Z

    V_SLICS_U8 ZERO, vshift, #4
    V_ADDICR_U8 vn, vn, #4
    V_SRICR_U8 vshift, vshift, #4

    V_ANDICS_8 ZERO, vshift, #0x3
    V_ADDICR_U8 vn, vn, #2
    V_SRICR_U8 vshift, vshift, #2

    V_ANDICS_8 ZERO, vshift, #0x1
    V_ADDICR_U8 vn, vn, #1

    V_SUBICS_U8 ZERO, _op, #0

    V_MV_8 _dst, vn

    V_MVICR_8 _dst, #8
.endmacro

/**
 * @brief Count trailing zeros on data type ushort4
 *
 * Macro to count trailing zeros on data type ushort4 called by __builtin_vmp_ctz_16(ushort4 x)
*/
.macro vmp_ctz_16(_dst, _op)
    V_MV_16 vshift, _op
    V_MVI_16 vn, #0

    MVI VCONDSEL, #COND_Z

    V_SLICS_U16 ZERO, vshift, #8
    V_ADDICR_U16 vn, vn, #8
    V_SRICR_U16 vshift, vshift, #8

    V_ANDICS_16 ZERO, vshift, #0xf
    V_ADDICR_U16 vn, vn, #4
    V_SRICR_U16 vshift, vshift, #4

    V_ANDICS_16 ZERO, vshift, #0x3
    V_ADDICR_U16 vn, vn, #2
    V_SRICR_U16 vshift, vshift, #2

    V_ANDICS_16 ZERO, vshift, #0x1
    V_ADDICR_U16 vn, vn, #1

    V_SUBICS_U16 ZERO, _op, #0

    V_MV_16 _dst, vn

    V_MVICR_16 _dst, #16
.endmacro

/**
 * @brief Count trailing zeros on data type uint2
 *
 * Macro to count trailing zeros on data type uint2 called by __builtin_vmp_ctz_32(uint2 x)
*/
.macro vmp_ctz_32(_dst, _op)
    V_MV_32 vshift, _op
    V_MVI_32 vn, #0

    MVI VCONDSEL, #COND_Z

    V_SLICS_U32 ZERO, vshift, #16
    V_ADDICR_U32 vn, vn, #16
    V_SRICR_U32 vshift, vshift, #16

    V_SLICS_U32 ZERO, vshift, #24
    V_ADDICR_U32 vn, vn, #8
    V_SRICR_U32 vshift, vshift, #8

    V_ANDICS_32 ZERO, vshift, #0xf
    V_ADDICR_U32 vn, vn, #4
    V_SRICR_U32 vshift, vshift, #4

    V_ANDICS_32 ZERO, vshift, #0x3
    V_ADDICR_U32 vn, vn, #2
    V_SRICR_U32 vshift, vshift, #2

    V_ANDICS_32 ZERO, vshift, #0x1
    V_ADDICR_U32 vn, vn, #1

    V_SUBICS_U32 ZERO, _op, #0

    V_MV_32 _dst, vn

    V_MVICR_32 _dst, #32
.endmacro


/**
 * @brief Population count on data type uchar8
 *
 * Macro to count number of non-0 bits on data type uchar8 called by __builtin_vmp_popcount_8(uchar8 x)
*/
.macro vmp_popcount_8(_dst, _op)
    MVI VCONDSEL, #COND_C
    V_MVI_8 mvtmp0, #0

    V_SLICS_8   mvtmp1, _op, #1
    V_ADDICR_8  mvtmp0, mvtmp0, #1

.loop cnt=[0..5]
    V_SLICS_8   mvtmp1, mvtmp1, #1
    V_ADDICR_8  mvtmp0, mvtmp0, #1
.endloop

    V_SLICS_8   mvtmp1, mvtmp1, #1
    V_ADDICR_8  _dst, mvtmp0, #1
.endmacro

/**
 * @brief Population count on data type ushort4
 *
 * Macro to count number of non-0 bits on data type ushort4 called by __builtin_vmp_popcount_16(ushort4 x)
 *
 * Counts each byte of _op in 8 SLICS_8/ADDICR_8 iterations (per-byte popcount, value 0..8 in each byte),
 * then folds the two byte popcounts of each ushort lane into the lane via V_MIX/V_ADD_16.
 * V_MIXR_8 picks the even-indexed bytes (low byte of each ushort) zero-extended to a ushort;
 * V_MIXL_8 picks the odd-indexed bytes (high byte of each ushort) the same way.
*/
.macro vmp_popcount_16(_dst, _op)
    MVI VCONDSEL, #COND_C
    V_MVI_8 mvtmp0, #0

    V_SLICS_8   mvtmp1, _op, #1
    V_ADDICR_8  mvtmp0, mvtmp0, #1

.loop cnt=[0..5]
    V_SLICS_8   mvtmp1, mvtmp1, #1
    V_ADDICR_8  mvtmp0, mvtmp0, #1
.endloop

    V_SLICS_8   mvtmp1, mvtmp1, #1
    V_ADDICR_8  mvtmp2, mvtmp0, #1

    V_MIXR_8    mvtmp3, VZERO, mvtmp2
    V_MIXL_8    mvtmp2, VZERO, mvtmp2
    V_ADD_16    _dst,   mvtmp2, mvtmp3
.endmacro

/**
 * @brief Population count on data type uint2
 *
 * Macro to count number of non-0 bits on data type uint2 called by __builtin_vmp_popcount_32(uint2 x)
 *
 * Same scheme as vmp_popcount_16: per-byte popcount, fold to ushort popcounts via V_MIX/V_ADD_16,
 * then fold the two ushort popcounts of each uint lane via V_MIX/V_ADD_32.
*/
.macro vmp_popcount_32(_dst, _op)
    MVI VCONDSEL, #COND_C
    V_MVI_8 mvtmp0, #0

    V_SLICS_8   mvtmp1, _op, #1
    V_ADDICR_8  mvtmp0, mvtmp0, #1

.loop cnt=[0..5]
    V_SLICS_8   mvtmp1, mvtmp1, #1
    V_ADDICR_8  mvtmp0, mvtmp0, #1
.endloop

    V_SLICS_8   mvtmp1, mvtmp1, #1
    V_ADDICR_8  mvtmp2, mvtmp0, #1

    V_MIXR_8    mvtmp3, VZERO, mvtmp2
    V_MIXL_8    mvtmp2, VZERO, mvtmp2
    V_ADD_16    mvtmp2, mvtmp2, mvtmp3

    V_MIXR_16   mvtmp3, VZERO, mvtmp2
    V_MIXL_16   mvtmp2, VZERO, mvtmp2
    V_ADD_32    _dst,   mvtmp2, mvtmp3
.endmacro


/**
 * @brief Macro to fill memory with a constant byte
 *
 * This macro is also used by compiler.
 * Pointer are word addresses.
 * Size n is passed in number of words.
 *
*/
//void *memset(void *s, int c, size_t n);
.macro memset(_src, _val, _size)
memset:
    ADDICS mstmp0, _size, #0

    MV  mstmp1, VFIR0 // save VFIR0

    MV  VFIR0, _src

    MV  mvtmp0, _val
    V_PERMI_8   mvtmp0, mvtmp0, #0

    BSSR_PNT skip, #COND_Z
memsetloop:
    V_STORE (VFIR0)+, mvtmp0
    ELOOPR  mstmp0, memsetloop

skip:
    MV  VFIR0, mstmp1 //restore VFIR0
.scheduling off
    NOP;NOP
.scheduling on

.endmacro


/**
 * @brief Macro to copy memory area
 *
 * This macro is also used by compiler.
 * Pointer are word addresses.
 * Size n is passed in number of words.
 *
*/
//void *memcpy(void *dest, const void *src, size_t n);
.macro memcpy(_dst, _src, _size)
memcpy:
    ADDICS mstmp0, _size, #0

    MV  mstmp1, SFIR0 // save SFIR0
    MV  mstmp2, SFIR1 // save SFIR1

    MV  SFIR0, _dst
    MV  SFIR1, _src

    BSSR_PNT skip, #COND_Z
memcpyloop:
    MV  (SFIR0)+, (SFIR1)+
    ELOOPR  mstmp0, memcpyloop

skip:
    MV  SFIR0, mstmp1 //restore SFIR0
    MV  SFIR1, mstmp2 //restore SFIR0
.scheduling off
    NOP;NOP
.scheduling on

.endmacro

/**
 * @brief Macro to compare memory areas
 *
 * This macro is also used by compiler.
 * Pointer are word addresses.
 * Size n is passed in number of words.
 *
*/
//int memcmp(const void *s1, const void *s2, size_t n);
.macro memcmp(_result, _dst, _src, _size)
memcmp:
    ADDICS  mstmp0, _size, #0

    MV  mstmp1, VFIR0 // save VFIR0
    MV  mstmp2, VFIR1 // save VFIR1

    MV  VFIR0, _dst
    MV  VFIR1, _src
    MVI _result, #-1

    BSSR_PNT skip, #COND_Z
memcmploop:
    V_LOAD      mvtmp0, (VFIR1)+
    V_SUBCS_8   mvtmp3, (VFIR0)+, mvtmp0
    V_MVICR_8   mvtmp4, #0

    BVSR_OR_PNT skip, #COND_NZ, #0b11111111

    ELOOPR  mstmp0, memcmploop


    MVI _result, #0

skip:
    MV  VFIR0, mstmp1 //restore VFIR0
    MV  VFIR1, mstmp2 //restore VFIR0
.scheduling off
    NOP;NOP
.scheduling on

.endmacro



/// interface to std memory functions


    .csection VMP_CL_STRING
/**
 * @brief Function to copy memory area
 *
 * Prototype:
 * __local void *memcpy(__byteaddress __local void *dest,
 *                      __byteaddress __local const void *src, size_t n);
 *
 * Size n is passed in number of words.
 *
*/
    .export _memcpy
_memcpy:
    ADDICS mstmp0, sparam2, #0

    // convert pointer to bytes to pointer to words
    SRI sparam0, sparam0, #3
    SRI sparam1, sparam1, #3

    MV SFIR0, sparam0
    MV SFIR1, sparam1

    // return in case of size 0
    BSSA_PNT lr, #COND_Z
memcpyloop:
    MV (SFIR0)+, (SFIR1)+
    ELOOPR mstmp0, memcpyloop

    // return
    jla zero, lr

/**
 * @brief Function to fill memory with a constant byte
 *
 * Prototype:
 * __local void *memset(__byteaddress __local void *str, int c, size_t n);
 *
 * Size n is passed in number of words.
 *
*/
    .export _memset
_memset:
    ADDICS mstmp0, sparam2, #0

    // convert pointer to bytes to pointer to words
    SRI sparam0, sparam0, #3

    //TODO: change to use of DMA memset transfer
    MV VFIR0, sparam0
    MV mvtmp0, sparam1
    V_PERMI_8 mvtmp0, mvtmp0, #0

    // return in case of size 0
    BSSA_PNT lr, #COND_Z
memsetloop:
    V_STORE (VFIR0)+, mvtmp0
    ELOOPR mstmp0, memsetloop

    // return
    jla zero, lr

/**
 * @brief Function to compare memory areas
 *
 * Prototype:
 * int memcmp(__byteaddress __local const void *str1,
 *            __byteaddress __local const void *str2, size_t n);
 *
 * Size n is passed in number of words.
 *
*/
    .export _memcmp
_memcmp:
    ADDICS mstmp0, sparam2, #0

    // convert pointer to bytes to pointer to words
    SRI sparam0, sparam0, #3
    SRI sparam1, sparam1, #3

    MV VFIR0, sparam0
    MV VFIR1, sparam1
    MVI sparam0, #-1

    // return -1 in case of size 0
    BSSA_PNT lr, #COND_Z
memcmploop:
    V_LOAD  mvtmp0, (VFIR1)+
    V_SUBCS_8 mvtmp3, (VFIR0)+, mvtmp0
    V_MVICR_8 mvtmp4, #0

    // return -1 in case of difference in any byte
    BVSA_OR_PNT lr, #COND_NZ, #0b11111111

    ELOOPR mstmp0, memcmploop

    // return 0 in case of all bytes equal
    MVI sparam0, #0

    // return
    jla zero, lr

    .endsection

/// @endcond
