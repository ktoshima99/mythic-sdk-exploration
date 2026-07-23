/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2026 videantis GmbH
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
 * FILENAME:      main_dsections.asm
 *
 * DESCRIPTION:   Main data sections for generated CNN code
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/


/* ========================================================================== *
 * Misc. variables                                                            *
 * ========================================================================== */
.dsection main
	.org 0x30
.export _md // make this memory region visible to C code via extern main_data_t md;
_md:
	.alloc int_msgbuf_base
	.alloc ext_msgbuf_base
	.alloc num_f__num_w             // |    num_f     |     num_w    |
	.alloc num_h__num_h2            // |    num_h     |    num_h/2   |

	.alloc out_l_base
	.alloc out_r_base
	.alloc out_l_base_plus1
	.alloc out_r_base_plus1
	.alloc out_l_base_plus2
	.alloc out_r_base_plus2
	.alloc out_l_base_plus4
	.alloc out_r_base_plus4

	.alloc aux_l_base
	.alloc aux_r_base
	.alloc aux_l_base_plus1
	.alloc aux_r_base_plus1

	.alloc softpad_rest
	.alloc core_id
	.alloc tmp
	.alloc dbgsync_vmpcode

	.alloc pad_value

	.alloc store_out_short_desc[6*msg.SHORT_DESCR_LEN]
	.alloc load_aux_short_desc[6*msg.SHORT_DESCR_LEN]
	.alloc dmaout_ext_upper32
	.alloc aux_ext_upper32

	.alloc save_sp_SFIR3
	.alloc save_lr_softpad_first
.endsection

/* ========================================================================== *
 * CDATA                                                                      *
 *  limited to 0x100 words (= 2 KB)                                           *
 * ========================================================================== */
.dsection cdata
	.org 0x100
.export _cdata // make this memory region visible to C code via extern cdata_t cdata;
_cdata:
	.alloc(0x100) cdata
.endsection


/* ========================================================================== *
 * Data section used in Average-Pooling Layer Code                            *
 * ========================================================================== */
.dsection avgp, dmem
.export _avgp // make this memory region visible to C code via extern avgp_data_t avgp;
_avgp:
	.alloc scaler_tab[21]
.endsection

/* ========================================================================== *
 * Data section used for performance evaluation                               *
 * Performance evaluation code is included when VMP_ENABLE_MEASUREMENTS is set*
 * ========================================================================== */
.dsection eval
    .equ DISABLE_VMP_PROCESS = { 1 << 0 }  // if flag is set, disable all NN routines
    .equ DISABLE_VMP_PRELOAD = { 1 << 1 }  // if flag is set, disable main_preload() and all NN routines
    .equ DISABLE_DMA_WGT     = { 1 << 2 }  // if flag is set, disable weight DMA transfer
    .equ DISABLE_DMA_CDT     = { 1 << 3 }  // if flag is set, disable cdata  DMA transfer
    .equ DISABLE_DMA_AUX     = { 1 << 4 }  // if flag is set, disable aux    DMA transfer
    .equ DISABLE_DMA_INP_BIT = {      5 }  // if flag is set, disable input  DMA transfer (flag > 5bit immediate)
    .equ DISABLE_DMA_PAD_BIT = {      6 }  // if flag is set, disable pad    DMA transfer (flag > 5bit immediate)
    .equ DISABLE_DMA_OUT_BIT = {      7 }  // if flag is set, disable output DMA transfer (flag > 5bit immediate)

.export _eval // make this memory region visible to C code via extern eval_data_t eval;
_eval:
	.alloc mode // read from GPDATA2 at initial sync, evaluated against flags above
.endsection
