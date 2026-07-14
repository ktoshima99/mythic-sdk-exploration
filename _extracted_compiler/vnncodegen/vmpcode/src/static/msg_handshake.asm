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
 * FILENAME:      msg_handshake.asm
 *
 * DESCRIPTION:   Simple Host->v-MP messaging routines using GPDATA0 register
 *                See msg.asm for message data structure definition
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

.dsection msg, dmem
.org auto
.alloc data[MSG_SIZE]
.endsection

.macro msg_init
	MV	SR_MSG_CNT, ZERO
.endmacro


.macro msg_recv
start:
	WAIT #{1 << 0}
	// load msg
wait_for_msg:

	//wait for change on GPDATA0
	WAIT #{VMP_WAIT_MODE_GPDATA | VMP_WAIT_GPDATA0}
	// clear event flag for GPDATA0
	MVIL VMP_ADDR_GPDFLAGS, #~VMP_WAIT_GPDATA0

	// get external message address from GPDATA0 (lower 32 bits of 64 bit address)
	MVIL STORE_HIGH, #0x100
	MV VR_TMP01, GPDATA0

msg_avail:
	V_STORE dma_descr.dma0.ext_byteaddr, VR_TMP01

	ORCS	ZERO, SR_MSG_CNT, SR_MSG_CNT
	BSSR	#dont_load_layermsg, #COND_NZ

// load layer_msg and return to start label of msg_recv macro
	MVIL STORE_HIGH, #layer_msg.data
	MVIL dma_descr.dma0.xf__int_wordaddr__length_int__length, #layer_msg.SIZE
	V_STORE dma_descr.dma0.compr__stride_int__count_3d__count, ONE
	V_STORE dma_descr.dma0.stride_3d__stride, ZERO
	.scheduling off
	.scheduling on
	MVI BIU_DMA_CTRL, #{0 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
	WAIT #{1 << 0}

	MV	 SR_MSG_CNT, layer_msg.data		// SR_MSG_CNT = |num_msg_templates|    num_parts    |
	EXPNDL16 SR_MSG_CNT, SR_MSG_CNT			// SR_MSG_CNT = num_parts

	// release message
	MV	 GPDATA0, ZERO

	ORCS	 ZERO, SR_MSG_CNT, SR_MSG_CNT		// num_parts == 0 indicates the end of the network
.if {defined(STANDALONE_HANDSHAKE) && {STANDALONE_HANDSHAKE != 0}} || {defined(STANDALONE_LOOP) && {STANDALONE_LOOP != 0}}
	BSSR	main.restart, #COND_Z     		// no more msgs: jump to restart
.else
	BSSR	main.exit, #COND_Z     			// no more msgs: jump to exit
.endif


	JLR	ZERO, start

// load msg (always uncompressed in non-standalone/handshake mode)
dont_load_layermsg:

	MVIL STORE_HIGH, #msg.data
	MVIL dma_descr.dma0.xf__int_wordaddr__length_int__length, #msg.SIZE
	V_STORE dma_descr.dma0.compr__stride_int__count_3d__count, ONE
	V_STORE dma_descr.dma0.stride_3d__stride, ZERO

	.scheduling off
	.scheduling on
load:
	MVI BIU_DMA_CTRL, #{0 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
	WAIT #{1 << 0}

exit:
	V_MVFIRI	VFIR_TMP2, #msg.data
	ADD		SR_MSG_CNT, N_ONE, SR_MSG_CNT
.endmacro

.macro msg_rel
	MV	GPDATA0, ZERO
.endmacro


