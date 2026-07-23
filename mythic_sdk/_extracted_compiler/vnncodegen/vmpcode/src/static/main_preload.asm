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
 * FILENAME:      main_preload.asm
 *
 * DESCRIPTION:   function main_preload(msg_t *) called from main routine
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

.csection main

// ushort8 main_preload(__wordaddress msg_t *)
// @return ushort8
//         vparam1 = | d.c. | SFIR_AUX_DMA_DESCR | auxBaseL | auxBaseR |
//         vparam0 = | outBaseL   |     outBaseR | inpBaseL | inpBaseR |

.export _main_preload

_main_preload:

.if defined(VMP_ENABLE_MEASUREMENTS)
	MVFIRI sidx1, #eval.mode
.endif

    MV VFIR_TMP2, sparam0
	MVI VCONDSEL, #COND_Z

	// Check message flags
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_FLAGS__INP_LOAD_REST_TOTAL_CNT
	V_LOAD		VR_TMP00, (VFIR_TMP)

	// set main.pad_value to
	// 0x00000000_00000000  if MSGDL_PAD_NEG128 is clear
	// 0x80808080_80808080  if MSGDL_PAD_NEG128 is set
	// Note: VCONDSEL==COND_Z
	V_MVI_8		VR_TMP01, #0x80
	V_ANDILCS_64	ZERO, VR_TMP00, #{msg.MSGFL_PAD_NEG128 << 8}
	V_MVCR_64	VR_TMP01, ZERO
	MV		main.pad_value, VR_TMP01

	// Extract various parameters
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_NUM_F__W__MPRES_H__NUM_H

	// extract 	|     numF |     numW |   mpResH |     numH |
	V_ADDEXPNDHI_U16 VR_TMP00, (VFIR_TMP), #0	// numF   | numW
	V_STORE		main.num_f__num_w, VR_TMP00
	V_ADDEXPNDLI_U16 VR_TMP01, (VFIR_TMP), #0	// mpResH | numH
	MV		SR_AUX_TOTAL_CNT, VR_TMP01	// numH
	MAXI		SR_AUX_TOTAL_CNT, SR_AUX_TOTAL_CNT, #2
	V_SRI_U64	 VR_TMP02, VR_TMP01, #32	//   0    | mpResH
	MV		 SR_RESH, VR_TMP02
	V_SRI_U64	 VR_TMP03, VR_TMP01, #1		// mpResH>>1|numH>>1|
	V_MIXR_32	VR_TMP03, VR_TMP01, VR_TMP03    //    numH  |numH>>1|
	V_STORE		 main.num_h__num_h2,   VR_TMP03

/* ========================================================================== */
	// check if wgts need to be loaded
	V_ADDFIRI	 VFIR_TMP, VFIR_TMP2, #MSG_OFS_WGTS_RSVD__LEN__CDATA_RSVD__LEN
	V_SRI_U64	 VR_TMP00, (VFIR_TMP), #32      // |       0      | rsvd |wgts_len |
	V_ORICS_16	 VR_TMP00, VR_TMP00, #0
	BVSR_AND	 do_not_load_wgts, #COND_Z, #0b00000010

	// SR_WGT_BASE = (msg.flags & MSGFL_WGTS_DMEM2) ? 0x400 : 0x1000
	MVI 		 SR_WGT_BASE, #0x400
	V_ADDFIRI	 VFIR_TMP, VFIR_TMP2, #MSG_OFS_FLAGS__INP_LOAD_REST_TOTAL_CNT
	V_LOAD		 VR_TMP01, (VFIR_TMP)
	MV		 SR_TMP0, VR_TMP01
	SRI		 SR_TMP0, SR_TMP0, #8
	ANDICS		 SR_TMP0, SR_TMP0, #msg.MSGFL_WGTS_DMEM2
	SLICR 		 SR_WGT_BASE, SR_WGT_BASE, #2 		// if CONDZ: SR_WGT_BASE=0x1000

	// setup DMA descr to load weights
	V_ADDFIRI	 VFIR_TMP, VFIR_TMP2, #MSG_OFS_WGTS_EXT_ADDR
	V_LOAD		 VR_TMP01, (VFIR_TMP)                   // |            ext_byteaddr|

	MV		 VR_TMP02, SR_WGT_BASE			// |  00000000 |   wgt_base |
	V_MIXR_32	 VR_TMP02, VR_TMP02, VR_TMP00		// |  wgt_base |     |wgtlen|
	V_MIXR_16	 VR_TMP02, ZERO, VR_TMP02
	V_STORE  {dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE}, VR_TMP01
	V_STORE  {dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE+1}, VR_TMP02
	V_STORE  {dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE+2}, ONE
	V_STORE  {dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE+3}, ZERO

.scheduling off
.scheduling on
.if !defined(VMP_ENABLE_MEASUREMENTS)
	MVI    BIU_DMA_CTRL, #{dmachn_default | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
.else
	ANDICS ZERO, (sidx1 @dmem), #eval.DISABLE_DMA_WGT
	MVICR    BIU_DMA_CTRL, #{dmachn_default | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
.endif
	WAIT	#{1 << dmachn_default}

/* ========================================================================== */
do_not_load_wgts:
	// check if cdata needs to be loaded
	V_ADDFIRI	 VFIR_TMP, VFIR_TMP2, #MSG_OFS_WGTS_RSVD__LEN__CDATA_RSVD__LEN
	V_ORICS_16	 VR_TMP00, (VFIR_TMP), #0 // |  rsvd |wgts_len||  rsvd |cdata_len
	BVSR_AND	 do_not_load_cdata, #COND_Z, #0b00000010


	MVI 		 SR_CDATA_BASE, #cdata.cdata

	// setup DMA descr to load CDATA
	V_ADDFIRI	 VFIR_TMP, VFIR_TMP2, #MSG_OFS_CDATA_EXT_ADDR
	V_LOAD		 VR_TMP01, (VFIR_TMP)                   // |            ext_byteaddr|

	MV		 VR_TMP02, SR_CDATA_BASE		// |  00000000 | cdata_base |
	V_MIXR_32	 VR_TMP02, VR_TMP02, VR_TMP00		// |cdata_base |     |cdtlen|
	V_MIXR_16	 VR_TMP02, ZERO, VR_TMP02
	V_STORE  {dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE}, VR_TMP01
	V_STORE  {dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE+1}, VR_TMP02
	V_STORE  {dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE+2}, ONE
	V_STORE  {dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE+3}, ZERO

.scheduling off
.scheduling on
.if !defined(VMP_ENABLE_MEASUREMENTS)
	MVI    BIU_DMA_CTRL, #{dmachn_default | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
.else
	ANDICS ZERO, (sidx1 @dmem), #eval.DISABLE_DMA_CDT
	MVICR    BIU_DMA_CTRL, #{dmachn_default | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
.endif
	WAIT	#{1 << dmachn_default}

/* ========================================================================== */

do_not_load_cdata:

// AUX
	// get aux reference external start address and inc
	// setup  SR_DMA_LOAD_AUX_ADDR, aux_ext_upper32 (!), SR_DMA_LOAD_AUX_INC (, SR_DMA_LOAD_AUX_INC_2)
	// from SG_OFS_AUX_LOAD_EXT_BASE      [                        load_aux_ext_base ]
	// from SG_OFS_AUX_LOAD_EXT_INC2__INC [     load_aux_inc2 |         load_aux_inc ]

	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_AUX_LOAD_EXT_BASE
	V_LOAD 		VR_TMP00, (VFIR_TMP)
	MV		SR_DMA_LOAD_AUX_ADDR, VR_TMP00
	MV		main.aux_ext_upper32, LOAD_HIGH

	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_AUX_LOAD_EXT_INC_2__INC
	V_LOAD		VR_TMP00, (VFIR_TMP)
	MV		SR_DMA_LOAD_AUX_INC, VR_TMP00
	ORCS		ZERO, SR_DMA_LOAD_AUX_ADDR, SR_DMA_LOAD_AUX_ADDR
	BSSR		no_aux, #COND_Z

	MV		SR_DMA_LOAD_AUX_INC_2, LOAD_HIGH
	ORCS		SR_DMA_LOAD_AUX_INC_2, SR_DMA_LOAD_AUX_INC_2, SR_DMA_LOAD_AUX_INC_2
	MVCR		SR_DMA_LOAD_AUX_INC_2, SR_DMA_LOAD_AUX_INC



// Auxiliary Input is currently used:
// - For merged shortcut layers to load the "far" shortcut branch
// - In special numL=4/numF=1/inpWW=1 mode
//   Only one DMA-Descriptor per row is being used in this mode, which is signalled
//   by setting msg.auxBaseR to ZERO
aux:
	// create 6 aux short descriptors (main.load_aux_short_desc[0..5]):
	//
	// Use aux_dmatemplate_load from message:
	// VR_TMP00 = |  xfer_flags|int_wordaddr|  length_int|      length|
	// VR_TMP01 = |  compr_mode|  stride_int|    count_3d|       count|
	// VR_TMP02 = |          stride_3d      |          stride         |
	//
	// Replace field "int_wordaddr" (VR_TMP00 Bits 47..32) with values from aux_load_int_addr[0..5]
	// and store result in main.load_aux_short_desc[0..5]
	//
	// pseudo code:
	// for(i = 0; i < 6; i++) {
	// 	tmp[0] = msg.short_desc_load_aux[i*3+0]
	// 	tmp[1] = msg.short_desc_load_aux[i*3+1]
	// 	tmp[2] = msg.short_desc_load_aux[i*3+2]
	//	tmp[0].int_wordaddr = aux_load_int_addr[i]
	//	main.load_aux_short_desc[i*3+0] = tmp[0]
	//	main.load_aux_short_desc[i*3+1] = tmp[1]
	//	main.load_aux_short_desc[i*3+2] = tmp[2]
	// }
	//

	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_AUX_DMATEMPLATE_LOAD
	V_LOAD		VR_TMP00, (VFIR_TMP)+
	V_LOAD		VR_TMP01, (VFIR_TMP)+
	V_LOAD		VR_TMP02, (VFIR_TMP)
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_AUX_LOAD_INT_ADDR0__ADDR1__ADDR2__ADDR3
	V_LOAD		VR_TMP03, (VFIR_TMP)

.if defined(VMP_ENABLE_MEASUREMENTS)
	// if AUX DMA transfer should be disabled, set length to zero
	// these transfers are triggered inside generated asm code
	MVI  SCONDSEL, #COND_NZ
	ANDICS ZERO, (sidx1 @dmem), #eval.DISABLE_DMA_AUX
	MVICR  VR_TMP00, #0
.endif

	MVIL		VPERMREG, #0x030b0100	// PERMREG mask to replace only int_wordaddr element
	MVFIRI		SFIR_AUX_DMA_DESCR, #main.load_aux_short_desc

	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP00
	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP01
	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP02
	MV		SR_TMP0, #3
aux_set_desc_loop1:
	V_SLI_64	VR_TMP03, VR_TMP03, #16
	V_PERMREG_16	VR_TMP00, VR_TMP00, VR_TMP03
	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP00
	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP01
	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP02
	ELOOPR 		SR_TMP0, #aux_set_desc_loop1

	MV		SR_TMP0, #2
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_AUX_LOAD_INT_ADDR4__ADDR5__BASE_L__R // ushort4 auxLoadIntAddr4_auxLoadIntAddr5_auxBaseL_auxBaseR;
	V_LOAD		VR_TMP03, (VFIR_TMP)
	V_MV		vparam1, VR_TMP03

aux_set_desc_loop2:
	V_PERMREG_16	VR_TMP00, VR_TMP00, VR_TMP03
	V_SLI_64	VR_TMP03, VR_TMP03, #16
	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP00
	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP01
	MV		(SFIR_AUX_DMA_DESCR)+, VR_TMP02
	ELOOPR 		SR_TMP0, #aux_set_desc_loop2
	MVFIRI		SFIR_AUX_DMA_DESCR, #main.load_aux_short_desc

	// adjust main.aux_load_in_short_desc[1,3,5], if dec_count[1] != 0
	// dec_cnt[1] is where you would expect int_addr[0], but int_addr[0] does not exist
	// Note: VCONDSEL == COND_Z
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_AUX_LOAD_INT_ADDR0__ADDR1__ADDR2__ADDR3
	V_SRICS_U64	VR_TMP00, (VFIR_TMP@dmem), #48					  //  0 | 0 | 0 |dec
	BVSR_OR		aux_no_adjust, #COND_Z, #0x80

	V_MVFIRI	VFIR_CDATA, #{main.load_aux_short_desc + 1 * msg.SHORT_DESCR_LEN}   // Use VFIR_CDATA as tmp
	MV		SR_TMP0, #3
aux_adjust_loop:
	V_LOAD		VR_TMP01, (VFIR_CDATA)+                                           // ...|...|...|len
	V_LOAD		VR_TMP02, (VFIR_CDATA)-                                           // ...|...|...|cnt
	V_SUB_16	VR_TMP02, VR_TMP02, VR_TMP00					  // ...|...|...|(cnt-dec)
	V_SLICS_U64	ZERO, VR_TMP02, #48						  // (cnt-dec == 0) ?
	V_MVCR_64	VR_TMP01, ZERO						          // yes: len=0
	V_STORE		(VFIR_CDATA)+, VR_TMP01
	V_STORE		(VFIR_CDATA)-, VR_TMP02
	V_ADDFIRI	VFIR_CDATA, VFIR_CDATA, #{2 * msg.SHORT_DESCR_LEN}   // desc[1] -> desc[3] -> desc[5]
	ELOOPR		SR_TMP0, aux_adjust_loop
aux_no_adjust:

	// setup aux local addresses
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_AUX_LOAD_INT_ADDR4__ADDR5__BASE_L__R // ushort4 auxLoadIntAddr4_auxLoadIntAddr5_auxBaseL_auxBaseR;
	V_ADDEXPNDLI_U16 VR_TMP00, (VFIR_TMP), #0
	V_SRI_U64	VR_TMP01, VR_TMP00, #32

	MV		main.aux_l_base, VR_TMP01
	MV		main.aux_r_base, VR_TMP00
	V_ADD_32	VR_TMP00, ONE, VR_TMP00
	V_ADD_32	VR_TMP01, ONE, VR_TMP01
	MV		main.aux_l_base_plus1, VR_TMP01
	MV		main.aux_r_base_plus1, VR_TMP00

	MV    SCONDSEL, #COND_NZ
	MV    SR_TMP0, main.aux_r_base	// aux_r_base
	ORCS  ZERO, SR_TMP0, SR_TMP0

	MV		SR_TMP1, SR_RESH
	SRICR		SR_TMP1, SR_TMP1, #1
	MV		SR_AUX_DMA_CNT, SR_TMP1

.if defined(VMP_ENABLE_MEASUREMENTS)
	ANDICS ZERO, (sidx1 @dmem), #eval.DISABLE_DMA_AUX
	BSSR no_aux, #COND_NZ
.endif

	MVI  SR_TMP0, #2
dma_preload_aux:
	MVI  SCONDSEL, #COND_P
	WAIT #WAIT_DMA_LOAD_AUX_MASK
	MV   STORE_HIGH, main.aux_ext_upper32

	MV              DMA_DESCR_LOAD_AUX0+0, SR_DMA_LOAD_AUX_ADDR
	ADD             SR_DMA_LOAD_AUX_ADDR, SR_DMA_LOAD_AUX_ADDR, SR_DMA_LOAD_AUX_INC
	MV              DMA_DESCR_LOAD_AUX0+1, (SFIR_AUX_DMA_DESCR)+
	MV              DMA_DESCR_LOAD_AUX0+2, (SFIR_AUX_DMA_DESCR)+
	MV              DMA_DESCR_LOAD_AUX0+3, (SFIR_AUX_DMA_DESCR)+
	ADDCS	    	SR_AUX_TOTAL_CNT, N_ONE, SR_AUX_TOTAL_CNT
.if defined(DISABLE_DMA_AUX)
	MVICR           ZERO, #DMACTRL_LOAD_AUX0_MASK
.else
	MVICR           BIU_DMA_CTRL, #DMACTRL_LOAD_AUX0_MASK
.endif
	// skip second dma_descr if main_aux_r_base==1 <-> auxBaseR==0
	V_SUBCS_32	ZERO, ONE, VR_TMP00
	BVSR_OR		skip_preload_aux_r, #COND_Z, #0b00001000

	MV              STORE_HIGH, main.aux_ext_upper32
	MV              DMA_DESCR_LOAD_AUX1+0, SR_DMA_LOAD_AUX_ADDR
	ADD             SR_DMA_LOAD_AUX_ADDR, SR_DMA_LOAD_AUX_ADDR, SR_DMA_LOAD_AUX_INC_2
	MV              DMA_DESCR_LOAD_AUX1+1, (SFIR_AUX_DMA_DESCR)+
	MV              DMA_DESCR_LOAD_AUX1+2, (SFIR_AUX_DMA_DESCR)+
	MV              DMA_DESCR_LOAD_AUX1+3, (SFIR_AUX_DMA_DESCR)+
	ADDCS	    	SR_AUX_TOTAL_CNT, N_ONE, SR_AUX_TOTAL_CNT
.if defined(DISABLE_DMA_AUX)
	MVICR           ZERO, #DMACTRL_LOAD_AUX1_MASK
.else
	MVICR           BIU_DMA_CTRL, #DMACTRL_LOAD_AUX1_MASK
.endif

skip_preload_aux_r:

	MVI SCONDSEL, #COND_Z

	ADDCS		SR_AUX_DMA_CNT, N_ONE, SR_AUX_DMA_CNT
	MVCR		SR_AUX_DMA_CNT, SR_TMP1
	MVCR		SFIR_AUX_DMA_DESCR, #main.load_aux_short_desc

	ELOOPR          SR_TMP0, dma_preload_aux

no_aux:
	MV VR_TMP00, SFIR_AUX_DMA_DESCR
	V_MIXR_32	vparam1, VR_TMP00, vparam1 // | d.c. | SFIR_AUX_DMA_DESCR | auxBaseL | auxBaseR |

	MVI SCONDSEL, #COND_Z

	// extract 	| inpBaseL | inpBaseR | softpadFirst | softPadRest |
	V_ADDFIRI	    VFIR_TMP, VFIR_TMP2, #MSG_OFS_INP_BASE_L__R__SOFTPAD_FIRST__REST    // ushort4 inpBaseL_inpBaseR_softpadFirst_softpadRest;
	V_SRI_U64 vparam0, (VFIR_TMP), #32
	// extract  |  d.c.    |    d.c.  |    outBaseL  |    outBaseR |
	V_ADDFIRI	    VFIR_TMP, VFIR_TMP2, #MSG_OFS_OUT_STORE_INT_ADDR4__ADDR5__BASE_L__R // ushort4 outStoreIntAddr4_outStoreIntAddr5_outBaseL_outBaseR;
	V_MIXR_32 vparam0, (VFIR_TMP), vparam0 // | outBaseL | outBaseR | inpBaseL | inpBaseR |


	// set main.out_*_base to outBase*+1 (for cyclic buffer reset)
	V_ADDEXPNDLI_U16 VR_TMP00, vparam0, #0	// inpBaseL | inpBaseR
	V_ADDEXPNDHI_U16 VR_TMP01, vparam0, #0	// outBaseL | outBaseR

	V_SRI_U64	VR_TMP02, VR_TMP00, #32
	V_SRI_U64	VR_TMP03, VR_TMP01, #32

	MV	main.out_r_base, VR_TMP01
	V_ADD_32 VR_TMP01, VONE, VR_TMP01
	MV	main.out_r_base_plus1, VR_TMP01
	V_ADDI_32 VR_TMP01, VR_TMP01, #1
	MV	main.out_r_base_plus2, VR_TMP01
	V_ADDI_32 VR_TMP01, VR_TMP01, #2
	MV	main.out_r_base_plus4, VR_TMP01

	MV	main.out_l_base, VR_TMP03
	V_ADD_32 VR_TMP03, VONE, VR_TMP03
 	MV	main.out_l_base_plus1, VR_TMP03
	V_ADDI_32 VR_TMP03, VR_TMP03, #1
 	MV	main.out_l_base_plus2, VR_TMP03
	V_ADDI_32 VR_TMP03, VR_TMP03, #2
	MV	main.out_l_base_plus4, VR_TMP03

.macro schedule_dma_pad_shortdesc(CHN, DESCR, PRIO)
	V_ADDFIRI  VFIR_TMP, VFIR_TMP2, #DESCR
	V_LOAD	   VR_TMP00, (VFIR_TMP)  // xf__int_wordaddr__length_int__length
	V_ORCS_16  VR_TMP00, VR_TMP00, VR_TMP00
	BVSR_PNT_AND  skip_dma_pad_shortdesc, #COND_Z, #0b00000010	// if length == 0 : skip memset

	MV  {dma_descr.start + CHN * dma_descr.CHANNELSIZE}, main.pad_value
	V_STORE  {dma_descr.start + CHN * dma_descr.CHANNELSIZE+1}, (VFIR_TMP@dmem)+
	V_STORE  {dma_descr.start + CHN * dma_descr.CHANNELSIZE+2}, (VFIR_TMP@dmem)+
	V_STORE  {dma_descr.start + CHN * dma_descr.CHANNELSIZE+3}, (VFIR_TMP@dmem)
.if !defined(VMP_ENABLE_MEASUREMENTS)
	MVI    BIU_DMA_CTRL, #{CHN | dma_descr.DMA_MEMSET | dma_descr.DMA_READ | PRIO}
.else
	SLI mstmp0, ONE, #eval.DISABLE_DMA_PAD_BIT
	ANDCS ZERO, (sidx1 @dmem), mstmp0
	MVICR  BIU_DMA_CTRL, #{CHN | dma_descr.DMA_MEMSET | dma_descr.DMA_READ | PRIO}
.endif
skip_dma_pad_shortdesc:
.endmacro

.macro schedule_dma_load(CHN, DESCR, PRIO)
	V_ADDFIRI  VFIR_TMP, VFIR_TMP2, #DESCR+1
	V_LOAD	   VR_TMP00, (VFIR_TMP)-  // xf__int_wordaddr__length_int__length
	V_ORCS_16  VR_TMP00, VR_TMP00, VR_TMP00
	BVSR_PNT_OR  skip_dma_load, #COND_Z, #0b00000010	// if length == 0 : skip load

	V_STORE  {dma_descr.start + CHN * dma_descr.CHANNELSIZE}, (VFIR_TMP@dmem)+
	V_STORE  {dma_descr.start + CHN * dma_descr.CHANNELSIZE+1}, (VFIR_TMP@dmem)+
	V_STORE  {dma_descr.start + CHN * dma_descr.CHANNELSIZE+2}, (VFIR_TMP@dmem)+
	V_STORE  {dma_descr.start + CHN * dma_descr.CHANNELSIZE+3}, (VFIR_TMP@dmem)
.if !defined(VMP_ENABLE_MEASUREMENTS)
	MVI    BIU_DMA_CTRL, #{CHN | dma_descr.DMA_READ | PRIO}
.else
	SLI mstmp0, ONE, #eval.DISABLE_DMA_INP_BIT
	ANDCS ZERO, (sidx1 @dmem), mstmp0
	MVICR  BIU_DMA_CTRL, #{CHN | dma_descr.DMA_READ | PRIO}
.endif
skip_dma_load:
.endmacro

	.call schedule_dma_pad_shortdesc(dmachn_pad_vertical,    MSG_OFS_DMA_DESCR_PADV,    dma_descr.DMA_PRIO_INT3)
	.call schedule_dma_pad_shortdesc(dmachn_pad_left,        MSG_OFS_DMA_DESCR_PADL,    dma_descr.DMA_PRIO_INT3)
	.call schedule_dma_load  (dmachn_load_inp_first,  MSG_OFS_DMA_DESCR_LOAD_INP_FIRST,  dma_descr.DMA_PRIO_INT2)
	.call schedule_dma_pad_shortdesc(dmachn_pad_right_first, MSG_OFS_DMA_DESCR_PADR_FIRST, dma_descr.DMA_PRIO_INT1)

 	WAIT #{{1 << dmachn_pad_vertical} | {1 << dmachn_pad_left} | {1 << dmachn_load_inp_first} | {1 << dmachn_pad_right_first}}

	.call schedule_dma_load  (dmachn_load_inp_rest,  	     MSG_OFS_DMA_DESCR_LOAD_INP_REST,  dma_descr.DMA_PRIO_INT2)
	.call schedule_dma_pad_shortdesc(dmachn_pad_right_rest,  MSG_OFS_DMA_DESCR_PADR_REST, dma_descr.DMA_PRIO_INT1)

/* ========================================================================== */
	// Soft-Padding
	// - if softpFirst is non-zero: call softpFirst routine
	// - setup softpRest routine to be invoked later when the entire input is available

	// extract | softpFirst | softpRest |
	V_ADDFIRI VFIR_TMP, VFIR_TMP2, #MSG_OFS_INP_BASE_L__R__SOFTPAD_FIRST__REST
	V_ADDEXPNDLI_U16 VR_TMP01, (VFIR_TMP), #0 // | softpFirst | softpRest  |
	V_SRICS_U64      VR_TMP02, VR_TMP01, #32  // |            | softpFirst |
	MV               SR_TMP0, VR_TMP02        //   softpFirst

	V_MIXR_32 VR_TMP03, ZERO, VR_TMP01        //   softpRest
	MV main.softpad_rest, VR_TMP03            //   softpRest


	// branch to softpadpFirst routine if softpFirst is non-zero
	MV  main.save_lr_softpad_first, lr
	MVI lr, #ret_from_softpad_first
.if defined(VMP_ENABLE_MEASUREMENTS)
	MVI SCONDSEL, #COND_NZ
	ANDICS ZERO, (sidx1 @dmem), #eval.DISABLE_VMP_PROCESS
	MVICR SR_TMP0, #ret_from_softpad_first
.endif
	BVSA_PNT_AND	 SR_TMP0, #COND_NZ, #0b10000000 // branch to softpFirst, if softpFirst != 0
ret_from_softpad_first:
	MV  lr, main.save_lr_softpad_first

/* ========================================================================== *
 * setup DMAOUT descriptors                                                   *
 * ========================================================================== */

	// setup  SR_DMAOUT_EXT_ADDR, dmaout_ext_upper32 (1), SR_DMAOUT_EXT_INC (, SR_DMAOUT_EXT_INC2)
	// from MSG_OFS_OUT_STORE_EXT_BASE, MSG_OFS_OUT_STORE_EXT_INC_2__INC
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_OUT_STORE_EXT_BASE
	V_LOAD		VR_TMP00, (VFIR_TMP)
	MV		SR_DMAOUT_EXT_ADDR, VR_TMP00
	MV		main.dmaout_ext_upper32, LOAD_HIGH


	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_OUT_STORE_EXT_INC_2__INC
	V_LOAD		VR_TMP00, (VFIR_TMP)
	MV		SR_DMAOUT_EXT_INC,  VR_TMP00
//	MV		SR_DMAOUT_EXT_INC_2, LOAD_HIGH

	// create 6 out short descriptors (main.store_out_short_desc[0..5]):
	//
	// Use out_dmatemplate_store from message:
	// VR_TMP00 = |  xfer_flags|int_wordaddr|  length_int|      length|
	// VR_TMP01 = |  compr_mode|  stride_int|    count_3d|       count|
	// VR_TMP02 = |          stride_3d      |          stride         |
	//
	// Replace field "int_wordaddr" (VR_TMP00 Bits 47..32) with values from out_store_int_addr[1..5]
	// and store result in main.store_out_short_desc[1..5]
	// Note main.store_out_short_desc[0].int_addr is not changed. "int_addr[0]" containes different data
	// (see msg.h)

	// pseudo code:
	// for(i = 1; i < 6; i++) {
	// 	tmp[0] = msg.short_desc_store_out[i*3+0]
	// 	tmp[1] = msg.short_desc_store_out[i*3+1]
	// 	tmp[2] = msg.short_desc_store_out[i*3+2]
	//	tmp[0].int_wordaddr = out_store_int_addr[i]
	//	main.store_out_short_desc[i*3+0] = tmp[0]
	//	main.store_out_short_desc[i*3+1] = tmp[1]
	//	main.store_out_short_desc[i*3+2] = tmp[2]
	// }
	//

	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_OUT_DMATEMPLATE_STORE

	V_LOAD		VR_TMP00, (VFIR_TMP)+	// |   xflags   |int_wordaddr| length_int |   length   |
	V_LOAD		VR_TMP01, (VFIR_TMP)+	// |   compr    | stride_int |  count3d   |    count   |
	V_LOAD		VR_TMP02, (VFIR_TMP)	// |         stride_3d       |         stride          |
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_OUT_STORE_INT_ADDR0__ADDR1__ADDR2__ADDR3
	V_LOAD		VR_TMP03, (VFIR_TMP)

.if defined(VMP_ENABLE_MEASUREMENTS)
	// if OUT DMA transfer should be disabled, set length to zero
	// these transfers are triggered inside generated asm code
	MVI  SCONDSEL, #COND_NZ
	SLI mstmp0, ONE, #eval.DISABLE_DMA_OUT_BIT
	ANDCS ZERO, (sidx1 @dmem), mstmp0
	MVICR  VR_TMP00, #0
.endif

	MVFIRI 		SFIR_OUT_DMA_DESCR, #main.store_out_short_desc

	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP00
	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP01
	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP02
	MV		SR_TMP0, #3
out_set_desc_loop1:
	V_SLI_64	VR_TMP03, VR_TMP03, #16
	V_PERMREG	VR_TMP00, VR_TMP00, VR_TMP03
	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP00
	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP01
	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP02
	ELOOPR SR_TMP0, #out_set_desc_loop1

	MV		SR_TMP0, #2
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_OUT_STORE_INT_ADDR4__ADDR5__BASE_L__R
	V_LOAD		VR_TMP03, (VFIR_TMP)
out_set_desc_loop2:
	V_PERMREG	VR_TMP00, VR_TMP00, VR_TMP03
	V_SLI_64	VR_TMP03, VR_TMP03, #16
	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP00
	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP01
	MV		(SFIR_OUT_DMA_DESCR)+, VR_TMP02
	ELOOPR SR_TMP0, #out_set_desc_loop2
	MVFIRI		SFIR_OUT_DMA_DESCR, #main.store_out_short_desc

	// adjust main.store_out_short_desc[1,3,5], if dec_count[1] != 0
	// dec_cnt[1] is where you would expect int_addr[0], but int_addr[0] does not exist
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_OUT_STORE_INT_ADDR0__ADDR1__ADDR2__ADDR3
	V_SRICS_U64	VR_TMP00, (VFIR_TMP@dmem), #48					  //  0 | 0 | 0 |dec
	BVSR_OR		out_no_adjust, #COND_Z, #0x80

	V_MVFIRI	VFIR_CDATA, #{main.store_out_short_desc + 1 * msg.SHORT_DESCR_LEN}   // Use VFIR_CDATA as tmp
	MV		SR_TMP0, #3
out_adjust_loop:
	V_LOAD   VR_TMP01, (VFIR_CDATA)+       // ...|...|...|len
	V_LOAD   VR_TMP02, (VFIR_CDATA)-       // ...|...|...|cnt
	V_SUB_16 VR_TMP02, VR_TMP02, VR_TMP00  // ...|...|...|(cnt-dec)
	V_SLICS_U64 ZERO, VR_TMP02, #48        // (cnt-dec == 0) ?
	V_MVCR_64   VR_TMP01, ZERO             //                  yes: len=0
	V_STORE		(VFIR_CDATA)+, VR_TMP01
	V_STORE		(VFIR_CDATA)-, VR_TMP02
	V_ADDFIRI	VFIR_CDATA, VFIR_CDATA, #{2 * msg.SHORT_DESCR_LEN}   // desc[1] -> desc[3] -> desc[5]
	ELOOPR		SR_TMP0, out_adjust_loop
out_no_adjust:


    JLA ZERO, lr

.endsection
