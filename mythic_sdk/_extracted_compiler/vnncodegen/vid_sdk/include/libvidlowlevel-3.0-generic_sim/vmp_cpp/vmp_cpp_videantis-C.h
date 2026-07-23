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
* FILENAME: vmp_cpp_videantis-C.h
*
* DESCRIPTION: videantis-C language mapping to C++
*              used to make builtins available
*              to static code analysis with FlexeLint and oclint
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef _VMP_CPP_VIDEANTISC_H_
#define _VMP_CPP_VIDEANTISC_H_

// connect to MISRA-C/C++
#include "vmp_cpp-misra.h"

// vector data types
#include "vmp_cpp-types.h"


#define  __builtin_vmp_convert_wordaddresstobyteaddress
#define  __builtin_vmp_convert_byteaddresstowordaddress


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



#endif /* _VMP_CPP_VIDEANTISC_H_ */
