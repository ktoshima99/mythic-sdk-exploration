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
 * FILENAME:      msg_standalone.asm
 *
 * DESCRIPTION:   Simple messaging routines for standalone mode
 *                See msg.asm for message data structure definition
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

.equ COMPRESS_MSGS = 1
.equ NEW_MSG_INIT = 1

// image input lo32 in GPDATA0
.equ GPDATA_IMG_INPUT_LO32_IDX  = 0
.equ GPDATA_IMG_INPUT_LO32      = GPDATA0

// image input hi32 in GPDATA1
.equ GPDATA_IMG_INPUT_HI32_IDX  = 1
.equ GPDATA_IMG_INPUT_HI32      = GPDATA1

// use GPDATA6 for inter-core synchronization
.equ GPDATA_INTER_CORE_SYNC_IDX = 6
.equ GPDATA_INTER_CORE_SYNC     = GPDATA6
.equ DCIF_VMP1_GPDATA_INTER_CORE_SYNC = VMP_CTRL_BASE_LO + VMP_MP0_REGWIN_BASE + {1 * VMP_REGWIN_SIZE} + GPDATA6_OFFSET

// use GPDATA7 as busy flag
.equ GPDATA_BUSY				      = GPDATA7

/// @brief void msg_load_next(uint2 internalAddr, long1 sizeWords)
/// load next message (SIZE=VR_TMP01 words) from external message buffer to local address INTADDR=VR_TMP00)
/// and increment byte address msg.cur_extaddr += sizeWords * 8
/// @param internalAddr data memory word address to load message buffer to (only s0 used)
/// @param sizeWords DMA transfer size in words
.csection loadMsg
.export _msg_load_next
_msg_load_next:
	V_MIXR_32 vparam0, vparam0, vparam1
	V_STORE   dma_descr.dma0.xf__int_wordaddr__length_int__length, vparam0
	MV        dma_descr.dma0.ext_byteaddr, msg.cur_extaddr

	V_SLI_64  vparam1, vparam1, #3 // SIZE=SIZE*8
	V_MVFIRI  VFIR_TMP, #msg.cur_extaddr
	V_ADD_32  (VFIR_TMP @dmem), (VFIR_TMP @dmem), vparam1

	V_STORE  dma_descr.dma0.compr__stride_int__count_3d__count, ONE
	V_STORE  dma_descr.dma0.stride_3d__stride, ZERO

	MVI BIU_DMA_CTRL, #{0 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
	WAIT #{1 << 0}

	JLA ZERO, lr
.endsection

/// @brief Initialize message reading at beginning of new image
/// 1. Initialize mutlicore synchronization
/// 2. Calculate cur_extaddr_init_addr = main.ext_msgbuf_base + main.core_id*sizeof(uint64_t)
///    and load msg.cur_extaddr from that address (DMA read single word at external address cur_extaddr_init_addr)
/// 3. BUILD MSGBUF START FROM INDEX
///    msg.cur_extaddr = msg.cur_extaddr + main.ext_msgbuf_base
.macro msg_init
.if defined(VMP_ENABLE_MEASUREMENTS)
	// DEBUG: frame counter in GPDATA3
	MV mstmp0, GPDATA3
	ADDI mstmp0, mstmp0, #1
	MV GPDATA3, mstmp0
.endif

	MVI	GPDATA_INTER_CORE_SYNC, #0

//FIXME: label required for outdated msgbase relocation
init_msgbase:

// NEW-IMAGE HANDSHAKE

.if defined(MODE_HOSTED) && {MODE_HOSTED != 0}

	// wait for new image
	WAIT #{VMP_WAIT_MODE_GPDATA | {1 << GPDATA_IMG_INPUT_LO32_IDX}}
	WAIT #{VMP_WAIT_MODE_GPDATA | {1 << GPDATA_IMG_INPUT_HI32_IDX}}

	// clear GPDFLAGS
	MVI	GPDFLAGS, #~{{1 << GPDATA_IMG_INPUT_LO32_IDX} | {1 << GPDATA_IMG_INPUT_HI32_IDX}}

.endif


// GET INDEX ADDRESS
  // load message buffer base from local memory
  V_LOAD		VR_TMP00, main.ext_msgbuf_base

	// Compute ext. address of init value for cur_extaddr, store in VR_TMP00
	// VR_TMP00 = [main.ext_msgbuf_base + core_id*sizeof(uint64_t)]
	V_MVFIRI   VFIR_TMP, #main.core_id
	V_SLI_U64  VR_TMP03, (VFIR_TMP @dmem), #3 // |          0           | core_id*sizeof(uint64_t) |
	V_ADD_32   VR_TMP00, VR_TMP00, VR_TMP03   // | main.ext_msgbuf_base + core_id*sizeof(uint64_t) |


// LOAD INDEX
	// Load initial value for msg.cur_extaddr from ext. address VR_TMP00
	// (computed in the previous step)
	V_STORE    dma_descr.dma0.ext_byteaddr, VR_TMP00
	MVIL 	   STORE_HIGH, #msg.cur_extaddr
	MV   	   dma_descr.dma0.xf__int_wordaddr__length_int__length, ONE
	V_STORE    dma_descr.dma0.compr__stride_int__count_3d__count, ONE
	V_STORE    dma_descr.dma0.stride_3d__stride, ZERO

	MVI BIU_DMA_CTRL, #{0 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
	WAIT #{1 << 0}

// BUILD MSGBUF START FROM INDEX
	V_LOAD   VR_TMP00, main.ext_msgbuf_base
	V_MVFIRI VFIR_TMP, #msg.cur_extaddr
	V_ADD_32 (VFIR_TMP @dmem), (VFIR_TMP @dmem), VR_TMP00

.if defined(COMPRESS_MSGS) && {COMPRESS_MSGS != 0}
	MV	SR_MSG_CNT, ZERO
.endif
.endmacro

/// @brief Receive next message
/// FIXME: add detailed documentation
.macro msg_recv
msg_avail:
.if defined(COMPRESS_MSGS) && {COMPRESS_MSGS != 0}

	//  while SR_MSG_CNT > 0 : process remaining messages
	//  otherwise load new layer_msg
	ORCS	ZERO, SR_MSG_CNT, SR_MSG_CNT
	BSSR	#dont_load_layermsg, #COND_NZ

	// load layer msg
load_layer_msg:
	// load next message (SIZE=vparam1 words) from external message buffer to local address INTADDR=vparam0)
	V_MVI_64 vparam1, #layer_msg.SIZE
	V_MVI_64 vparam0, #layer_msg.data
	JLR lr, _msg_load_next


	V_LOAD	VR_TMP00, layer_msg.data	// VR_TMP00 = | msgbuf_base| cdata_base |num_msg_tpls|  num_parts |
	MV	SR_TMP0,  VR_TMP00      	// SR_TMP0  = |num_msg_templates|    num_parts    |

	V_SRI_U64 VR_TMP00, VR_TMP00, #48	// VR_TMP00 = msgbuf_base
	MV	  main.int_msgbuf_base, VR_TMP00

	// init SR_MSG_CNT with num_parts
	EXPNDL16 SR_MSG_CNT, SR_TMP0

	// a layer_msg with num_parts == 0 indicates no more msgs (= end of network)
	ORCS	ZERO, SR_MSG_CNT, SR_MSG_CNT

.if {{defined(MODE_HOSTED) && {MODE_HOSTED != 0}}}
	MVI SCONDSEL, #COND_Z

	/// uint2 vid_prof_stop(void);
	/// Function to stop and read both TIMER registers
	/// returns two profiling 32Bit values in a uint2 vector word
	/// .s1 = all cycles when v-MP was in WAIT state since vmp_prof_start()
	/// .s0 = all cycles since vmp_prof_start()
	MVI lr, #_vid_prof_stop_return
	BSSR _vid_prof_stop, #COND_Z  // no more msgs: stop timers
_vid_prof_stop_return:
	// v-MP signals processing cycles back in GPDATA0 (total cycles 32 Bits) and GPDATA1 (wait cycles 32 Bits)
	MVCR GPDATA0, vparam0
	MVCR GPDATA1, LOAD_HIGH

	// hosted mode
	// if no more messages: clear busy flag
	MVICR	GPDATA_BUSY, #0
	// if no more messages: return to main.restart to re-enter loop waiting for new input
	BSSR	main.restart, #COND_Z  // no more msgs: jump to main.restart

.else
	// otherwise
	// if no more mesages: jump to exit label
	BSSR	main.exit, #COND_Z     // no more msgs: normal mode: jump to exit

.endif

	// extract num_msg_templates
	SRICS_U	SR_TMP0, SR_TMP0, #16
	// num_msg_templates == 0 indicates uncompressed-mode
	BSSR	#load_uncompressed, #COND_Z

	// compute num_msg_templates * msg.SIZE
	MVI        STORE_HIGH, #0
	MV         vparam1, SR_TMP0
	V_MULIL_32 vparam1, vparam1, #msg.SIZE
	// use computed value to load <num_msg_templates> message templates
	V_LOAD     vparam0, main.int_msgbuf_base
	JLR lr, _msg_load_next

	JLR	ZERO, #load_compressed

dont_load_layermsg:
	// check if uncompressed or compressed
	MV	SR_TMP0, layer_msg.data
	SRICS_U	SR_TMP0, SR_TMP0, #16
	BSSR	#load_uncompressed, #COND_Z

load_compressed:
	// load part_diff_msg
	V_MVI_64 vparam1, #PART_DIFF.SIZE
	V_MVI_64 vparam0, #PART_DIFF.MSG
	JLR lr, _msg_load_next

	// current message template base address is VFIR_TMP2 = msg.start + template_id * msg.SIZE
	V_LOAD		VR_PART_DIFF_MSG_3, PART_DIFF.MSG + PART_DIFF.WGTS_LEN__CDATA_LEN__IDX__TEMPLATE_ID
	V_MULIL_U16	VR_PART_DIFF_MSG_3, VR_PART_DIFF_MSG_3, #msg.SIZE
	V_LOAD		VR_TMP00, main.int_msgbuf_base
	V_ADD_16	VR_PART_DIFF_MSG_3, VR_PART_DIFF_MSG_3, VR_TMP00
	MV		VFIR_TMP2, VR_PART_DIFF_MSG_3

  // copy out_store_ext_base from part_diff_msg to msg
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_OUT_STORE_EXT_BASE
	V_LOAD		(VFIR_TMP), PART_DIFF.MSG + PART_DIFF.OUT_STORE_EXT_BASE
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_AUX_LOAD_EXT_BASE
	V_LOAD		(VFIR_TMP), PART_DIFF.MSG + PART_DIFF.AUX_LOAD_EXT_BASE

  // copy load_inp_first_ext_base from part_diff_msg to msg
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_DMA_DESCR_LOAD_INP_FIRST
	V_LOAD		(VFIR_TMP), PART_DIFF.MSG + PART_DIFF.LOAD_INP_FIRST_EXT_BASE
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_DMA_DESCR_LOAD_INP_REST
	V_LOAD		(VFIR_TMP), PART_DIFF.MSG + PART_DIFF.LOAD_INP_REST_EXT_BASE

  // copy part_diff_msg.wgts_ext_addr from part_diff_msg to msg
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_WGTS_EXT_ADDR
	V_LOAD		(VFIR_TMP), PART_DIFF.MSG + PART_DIFF.WGTS_EXT_ADDR
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_CDATA_EXT_ADDR
	V_LOAD		(VFIR_TMP), PART_DIFF.MSG + PART_DIFF.CDATA_EXT_ADDR

  // extract cdata_len, wgts_len from part_diff_msg to msg
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_WGTS_RSVD__LEN__CDATA_RSVD__LEN
	V_LOAD		VR_TMP00, PART_DIFF.MSG + PART_DIFF.WGTS_LEN__CDATA_LEN__IDX__TEMPLATE_ID
	V_ADDEXPNDH_U16	(VFIR_TMP), ZERO, VR_TMP00

  // extract debug_sync word from part_diff_msg to msg
	V_ADDFIRI	VFIR_TMP, VFIR_TMP2, #MSG_OFS_BUILD_TIMESTAMP__DEBUG_SYNC__VMP_CODE
	V_LOAD	  VR_TMP01, (VFIR_TMP)
  V_SELECTLE_16 VR_TMP01, VR_TMP00, #2
  V_STORE   (VFIR_TMP), VR_TMP01

JLR	ZERO, #exit


load_uncompressed:
	// load uncompressed message to msg.start
	V_LOAD   vparam0, main.int_msgbuf_base
	V_MVI_64 vparam1, #msg.SIZE
	MV       VFIR_TMP2, vparam0
	JLR lr, _msg_load_next

.else	/* if defined(COMPRESS_MSGS) && {COMPRESS_MSGS != 0} ... */

	// COMPRESS_MSGS macro undefined: just load (uncompressed) message
	V_LOAD   vparam0, main.int_msgbuf_base
	V_MVI_64 vparam1, #msg.SIZE
	MV       VFIR_TMP2, vparam0
	JLR lr, _msg_load_next

.endif	/* if defined(COMPRESS_MSGS) && {COMPRESS_MSGS != 0} ... else ...*/

exit:
.if defined(COMPRESS_MSGS) && {COMPRESS_MSGS != 0}
	// decrease message counter
	ADD	SR_MSG_CNT, N_ONE, SR_MSG_CNT
.endif
.endmacro

/// @brief Release message
/// FIXME: add detailed documentation
.macro msg_rel

// multi-core synchronization

	// only sync if debug/sync field is non-zero
	MV	SR_TMP0, main.dbgsync_vmpcode
	SRICS	SR_TMP0, SR_TMP0, #16
	BSSR	exit, #COND_Z

	// branch to slave sync, if debug/sync field is -1
	SUBCS	ZERO, N_ONE, SR_TMP0
	BSSR	slave_sync, #COND_Z

// master_sync


	WAIT #0xffff

// wait until all slaves are finished and waiting

	V_MVIL_32 VR_TMP00, #{VMP_CTRL_BASE_HI}, #0x80
	V_MVIL_32 VR_TMP00, #{DCIF_VMP1_GPDATA_INTER_CORE_SYNC}, #0x08
	MVIL 	  STORE_HIGH, #main.tmp
	MV   	  dma_descr.dma0.xf__int_wordaddr__length_int__length, ONE
	V_STORE   dma_descr.dma0.compr__stride_int__count_3d__count, ONE
	V_STORE   dma_descr.dma0.stride_3d__stride, ZERO

master_wait_slave:
	V_STORE   dma_descr.dma0.ext_byteaddr, VR_TMP00
	.scheduling off
	.scheduling on
	MVI BIU_DMA_CTRL, #{0 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
	WAIT #{1 << 0}

	SLEEP 16

	MV		VR1, main.tmp
	V_ORCS_32	VR1, VR1, VR1
	BVSR_AND	master_wait_slave, #COND_Z, #0b10000000

	V_ADDIL_32 VR_TMP00, VR_TMP00, #VMP_REGWIN_SIZE
	V_MVIL_32 VR_TMP00, #{VMP_CTRL_BASE_HI}, #0x80

	ELOOPR	SR_TMP0, master_wait_slave

// release all slaves
	MV	  SR_TMP0, main.dbgsync_vmpcode
	SRI	  SR_TMP0, SR_TMP0, #16
	MV	  STORE_HIGH, ZERO
	MV 	  main.tmp, ZERO
	V_MVIL_32 VR_TMP00, #{VMP_CTRL_BASE_HI}, #0x80
	V_MVIL_32 VR_TMP00, #{DCIF_VMP1_GPDATA_INTER_CORE_SYNC}, #0x08
master_release_slave:
	V_STORE   dma_descr.dma0.ext_byteaddr, VR_TMP00
	.scheduling off
	.scheduling on
	MVI BIU_DMA_CTRL, #{0 | dma_descr.DMA_WRITE | dma_descr.DMA_PRIO_STD}
	WAIT #{1 << 0}
	V_ADDIL_32 VR_TMP00, VR_TMP00, #VMP_REGWIN_SIZE
	V_MVIL_32 VR_TMP00, #{VMP_CTRL_BASE_HI}, #0x80

	ELOOPR	SR_TMP0, master_release_slave

	JLR	ZERO, #exit

//slave_sync

slave_sync:
	MVI	GPDATA_INTER_CORE_SYNC, #1

slave_wait:
	WAIT	#{VMP_WAIT_MODE_GPDATA | {1 << GPDATA_INTER_CORE_SYNC_IDX}}
	MVI		GPDFLAGS, #~{1 << GPDATA_INTER_CORE_SYNC_IDX}
	MV		SR_TMP0, GPDATA_INTER_CORE_SYNC
	ORCS	SR_TMP0, SR_TMP0, SR_TMP0
	BSSR	slave_wait, #COND_NZ

exit:
.endmacro

.dsection msg
	.org auto
	.alloc cur_extaddr
.endsection
