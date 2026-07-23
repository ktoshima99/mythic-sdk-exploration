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
* FILENAME: vid_vsp_cbam.h
*
* DESCRIPTION: bitstream read access macros
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/


#ifndef _VID_SP2_BITSTREAM_H
#define _VID_SP2_BITSTREAM_H

#define vid_get_bits_signed(VAL,W)  __asm__ volatile ("mvs_cc %0,  (-A14+)[0:" #W "]\n\t" : "=Rx" (VAL))
#define vid_get_bits(VAL,W)         __asm__ volatile ("mvu_cc %0,  (-A14+)[0:" #W "]\n\t" : "=Rx" (VAL))
#define vid_show_bits(VAL,W)        __asm__ volatile ("mvu_cc %0,   (A14) [0:" #W "]\n\t" : "=Rx" (VAL))
#define vid_flush_bits(W)           __asm__ volatile ("mvu_cc zero,(-A14+)[0:" #W "]\n\t")

#define vid_get_bsptr(VAL)          __asm__ volatile ("mv_ww %0,A14\n\t"    \
                                                  "NOP\n\t" \
                                                  : "=Rx" (VAL))

#if 0
#define vid_get_vbits_signed(VAL,W) VAL = __builtin_mvs_wv_a14_mp(W)
#define vid_get_vbits(VAL,W)  VAL = __builtin_mvu_wv_a14_mp(W)
#define vid_show_vbits(VAL,W) VAL = __builtin_mvu_wv_a14(W)
#define vid_flush_vbits(W) __builtin_mvu_wv_a14_mp(W)
#else
#define vid_get_vbits_signed(VAL,W) __asm__ volatile ("NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "mvs_wv %0,  (-A14+)[0:%1]\n\t" \
                                                  : "=Rx" (VAL): "BLRx" (W): "A14")
#define vid_get_vbits(VAL,W)        __asm__ volatile ("NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "mvu_wv %0,  (-A14+)[0:%1]\n\t" \
                                                  : "=Rx" (VAL): "BLRx" (W): "A14")
#define vid_show_vbits(VAL,W)       __asm__ volatile ("NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "mvu_wv %0,   (A14) [0:%1]\n\t" \
                                                  : "=Rx" (VAL): "BLRx" (W) )
#define vid_flush_vbits(W)          __asm__ volatile ("NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "NOP\n\t"             \
                                                  "mvu_wv zero,(-A14+)[0:%0]\n\t" \
                                                  :           : "BLRx" (W): "A14")
#endif

#define vid_get_moreDataFlag(OFFSET,LASTBITS)     __asm__ volatile ("mv     %0,A14\n\t" \
                                                                "mvu_cc %1, (A14) [0:8]\n\t" \
                                                                "NOP\n\t" \
                                                                "NOP\n\t" \
                                                                "NOP\n\t" \
                                                                : "=Rx" (OFFSET), "=Rx" (LASTBITS)  )

#define vid_ue_v(VAL) \
  { register unsigned BLRtmp; \
    __asm__ volatile ("MVU_CC          %1,(A14)             // CMB can not access the misaligned bitstream\n\t" \
                  "CMB_WWW         %1,ZERO,%1           // number of leading 0s\n\t" \
                  "LSL_WWW         %0,ONE,%1\n\t" \
                  "ADD_WI16        %0,-1                // (2^lz)-1\n\t" \
                  "NOP\n\t" \
                  "MVU_CC          ZERO,(-A14+)[0:1]    // consume first bit of prefix\n\t" \
                  "MVU_WV          ZERO,(-A14+)[0:%1]   // consume rest of prefix\n\t" \
                  "ADDU_WV_$       %0,(-A14+)[0:%1]     // consume and add suffix\n\t" \
                  : "=Rx" (VAL), "=BLRx" (BLRtmp) );}

#define vid_se_v(VAL) \
    __asm__ volatile (".if {1}\n\t" \
                  "MVU_CC          R24,(A14)            // CMB can not access the misaligned bitstream\n\t" \
                  "CMB_WWW_$       R24,ZERO,R24         // number of leading 0s\n\t" \
                  "ADD_WWW         %0,MONE,R24          // number of leading 0s - 1\n\t" \
                  "MV_WI16_Z       PC,#end\n\t" \
                  " LSL_WWW        %0,ONE,%0            // 2^(lz-1)\n\t" \
                  " MV_WW_Z        %0,ZERO\n\t" \
                  " MVU_CC         ZERO,(-A14+)[0:1]    // consume first bit of prefix\n\t" \
                  " MVU_WV         ZERO,(-A14+)[0:R24]  // consume rest of prefix\n\t" \
                  "MVU_WV_$        R24,(-A14+)[0:R24]   // consume and read suffix\n\t" \
                  "ADDU_CC         %0,R24[1:31]\n\t" \
                  "SUB_WI32W_Y     %0,#0,%0\n\t" \
                  "end:\n.endif\n\t" \
                  : "=Rx" (VAL) : : "R24");

#define __vld_noesc(VAL,TAB) \
    __asm__ volatile ("mv a8,#vlc_tables.vlctab_" #TAB "\n\t" \
                  "nop\n\t" \
                  "nop\n\t" \
                  "nop\n\t" \
                  "vld %0,(a8),(a14),vlctab_" #TAB "_entry_format,vlctab_" #TAB "_entry_length,0\n\t" \
                  : "=Rx" (VAL) : : "A8" , "A14")

/*
//does not work with current GCC version!
#define __vld_esc(VAL,TAB) \
    __asm__ volatile ("mv a8,#vlc_tables.vlctab_" #TAB "\n\t" \
                  "nop\n\t" \
                  "nop\n\t" \
                  "nop\n\t" \
                  "vld r0,(a8),(a14),vlctab_" #TAB "_entry_format,vlctab_" #TAB "_entry_length,_vlctab_" #TAB "_esc\n\t" \
                  : "=R0" (VAL) : : "R31", "A8", "A14")
*/

#endif
