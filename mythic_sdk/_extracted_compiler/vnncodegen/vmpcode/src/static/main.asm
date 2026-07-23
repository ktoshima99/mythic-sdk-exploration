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
 * FILENAME:      main.asm
 *
 * DESCRIPTION:   Main routine for generated CNN code
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// SDK compatible link register alias defined in vid_vmp_boot_loader.asm
.equ lr                     = SR31

// Main code
.csection main
	.export _main
	.org auto

_main:

// Initial host synchronization after booting
	// wait for pointer to ext. msgbuf in GPDATA0(lo32) and GPDATA1(hi32)
	WAIT  #{VMP_WAIT_MODE_GPDATA | VMP_WAIT_GPDATA1}
	MV    STORE_HIGH, GPDATA1
	WAIT  #{VMP_WAIT_MODE_GPDATA | VMP_WAIT_GPDATA0}
	MV     main.ext_msgbuf_base, GPDATA0

	// wait for core_id in GPDATA6
	WAIT  #{VMP_WAIT_MODE_GPDATA | VMP_WAIT_GPDATA6}
	MV STORE_HIGH, ZERO
	MV main.core_id, GPDATA6

	// save performance eval mode provided in GPDATA2
.if defined(VMP_ENABLE_MEASUREMENTS)
	MV eval.mode, GPDATA2
.endif

	// clear event flags for GPDATA 0, 1, 6
	MVI GPDFLAGS, #~{VMP_WAIT_GPDATA0 | VMP_WAIT_GPDATA1 | VMP_WAIT_GPDATA6}

	// signal back to host that we consumed the information
	// set GPDATA0 to -1
	MV GPDATA0, N_ONE


restart:
	.call msg_init

	// initialize write-shift registers to one
	V_MV_8 VR_TMP00, ONE
	V_STORE VACCU_SHIFT_WR+0, VR_TMP00
	V_STORE VACCU_SHIFT_WR+1, VR_TMP00
	V_STORE VACCU_SHIFT_WR+2, VR_TMP00
	V_STORE VACCU_SHIFT_WR+3, VR_TMP00

	// clear accu
	V_STORE VACCU_N_SHORT+0, ZERO
	V_STORE VACCU_N_SHORT+1, ZERO
	V_STORE VACCU_N_SHORT+2, ZERO
	V_STORE VACCU_N_SHORT+3, ZERO

	// call vid_prof_start(void);
	/// Function to stop and initialize both TIMER registers
	/// TIMER1 configured to count all cycles
	/// TIMER2 configured to count all cycles when in running state (e.g. no DMA wait)
	JLR lr, _vid_prof_start

mainloop:

/* ========================================================================== *
 *  Receive message                                                           *
 *    Parameters stored in: TODO                                              *
 *    Return parameters to: VFIR_TMP2                                         *
 * ========================================================================== */
	// receive message
	.call msg_recv

parse_msg:
	// parse msg
	MVI  VCONDSEL, #COND_Z
	MVI  SCONDSEL, #COND_Z
	MVIL VPERMREG, #0x030b0100

	// Check if vmpcode==0 -> indicating end of messages
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_BUILD_TIMESTAMP__DEBUG_SYNC__VMP_CODE
	V_LOAD		VR_TMP00, (VFIR_TMP)
	MV		main.dbgsync_vmpcode, VR_TMP00
	V_ORCS_16	ZERO, VR_TMP00, VR_TMP00
.if {defined(STANDALONE_HANDSHAKE) && {STANDALONE_HANDSHAKE != 0}} || {defined(STANDALONE_LOOP) && {STANDALONE_LOOP != 0}}
	BVSR_AND_PNT	restart, #COND_Z, #0b0000010  // no more msgs: standalone-mode-hs/loop: jump to restart
.else
	BVSR_AND_PNT	exit, #COND_Z, #0b0000010   // no more msgs: normal mode: jump to exit
.endif
	// vmpcode==0xffff indicates an idle msg
	V_SUBCS_16	ZERO, N_ONE, VR_TMP00
	BVSR_AND_PNT	next_msg, #COND_Z, #0b0000010   // idle msg: jump to next_msg

	// OVL
	// check whether to load code overlay
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_OVL_EXTSRC__INTDST__LEN
	V_ORICS_64	VR_TMP00, (VFIR_TMP), #0
	BVSR_AND_PNT	no_code_ovl, #COND_Z, #0b10000000

/* ========================================================================== */
	// FIXME: replace by overlay manager preload function
	V_MVFIRI	VFIR_TMP, #dma_descr.start + dmachn_default * dma_descr.CHANNELSIZE
	V_LOAD		VR_TMP01, boot.boot_address
	V_SRI_U64	VR_TMP02, VR_TMP00, #32
	V_ADD_U32	(VFIR_TMP)+, VR_TMP01, VR_TMP02	// |                    ext_addr |
	V_ADDEXPNDLI_U16 (VFIR_TMP)+, VR_TMP00, #0	// |      int_addr |         len |
	V_STORE		(VFIR_TMP)+, ZERO
	V_STORE		(VFIR_TMP)+, ZERO

.scheduling off
.scheduling on
	MVI    BIU_DMA_CTRL, #{dmachn_default | dma_descr.DMA_READ | DMA_PRIO_STD}
	WAIT	#{1 << dmachn_default}

/* ========================================================================== */
no_code_ovl:
	// call ushort8 main_preload(__wordaddress msg_t *) implemented in main_preload.asm
	MV sparam0, VFIR_TMP2

.if defined(VMP_ENABLE_MEASUREMENTS)
	MVI mstmp0, #get_routine_jump_address
	MVFIRI sidx0, #eval.mode
	ANDICS ZERO, (sidx0 @dmem), #eval.DISABLE_VMP_PRELOAD
	MVICR mstmp0, #_main_preload
	JLA lr, mstmp0
.else
	JLR lr, _main_preload
.endif
/* ========================================================================== */
get_routine_jump_address:
	// Get vmp_code routine jump address
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_BUILD_TIMESTAMP__DEBUG_SYNC__VMP_CODE
	V_LOAD		VR_TMP00, (VFIR_TMP)
	MV		SR_TMP0, VR_TMP00	// load vmp_code

	V_ORCS_32	ZERO, VR_TMP00, VR_TMP00
	BVSR_AND	exit, #COND_Z, #0x08


	MVI        SCONDSEL, #COND_Z     // TODO: remove "MV CONDSEL, #COND_Z" from generated routines
	MVIL       VPERMREG, #0x6420eca8

/* ========================================================================== *
 *  Jump to v-MP routine                                                      *
 *    Parameters stored in: TODO                                              *
 *    return parameters in ushort8 from main_preload()                        *
 * ========================================================================== */
.if defined(VMP_ENABLE_MEASUREMENTS)
	MVI mstmp0, #return_from_routine
	MVFIRI sidx0, #eval.mode
	ANDICS ZERO, (sidx0 @dmem), #{eval.DISABLE_VMP_PRELOAD | eval.DISABLE_VMP_PROCESS}
	MVCR mstmp0, SR_TMP0
	MV SR_TMP0, mstmp0
.endif

	// prepare return parameters in ushort8 from main_preload() into SFIR/VFIR registers

	//  vparam0 = | outBaseL | outBaseR | inpBaseL | inpBaseR |
	MV VFIR_INP_R, vparam0
	MV VFIR_OUT_R, LOAD_HIGH
	V_SRI_U32 vparam0, vparam0, #16
	MV VFIR_INP_L, vparam0
	MV VFIR_OUT_L, LOAD_HIGH

	// vparam1 = | d.c. | SFIR_AUX_DMA_DESCR | auxBaseL | auxBaseR |
	MV main.save_sp_SFIR3, SFIR3 // save stackpointer and restore after JLA lr
	MV SFIR_AUX_R, vparam1       // same as stackpointer!
	MV SFIR_AUX_DMA_DESCR, LOAD_HIGH
	V_SRI_U32 vparam1, vparam1, #16
	MV SFIR_AUX_L, vparam1

	MVFIRI SFIR_OUT_DMA_DESCR, #main.store_out_short_desc

	JLA	   lr, SR_TMP0
return_from_routine:

	// restore stackpointer
	MV SFIR3, main.save_sp_SFIR3

	WAIT #0xffff
next_msg:
	.call msg_rel
	JLR	ZERO, mainloop

/* ========================================================================== */
	// stop vidsim simulation using stop breakpoint
	// -bp stop=1 main.exit
exit:
	WAIT #0xffff
	.call msg_rel

	BREAK

.endsection
