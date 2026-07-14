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
 written authorization of videantis GmbH or an individual license agreementp
 with videantis GmbH is strictly forbidden.

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
*
* FILENAME: vid_vsp_cbam_read.asm
*
* DESCRIPTION: bitstream cyclic buffer read functions
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// NOTE: these functions use DMA channel 0 exclusively (channel descriptor must not be modified from outside)

.equ BITSTREAM_INBUF_SIZE	= 0x40w

// arguments:
//  r0: bitstream base address (external byte address)
.csection vid_bitstream_init_read
	.scheduling off
	.export	_vid_bitstream_init_read
_vid_bitstream_init_read:
	and_wwi32	IRQ,IRQ,#~IRQ_ENABLE_CBAM_BCHANGE		// disable CBAM interrupt

wait_prev_dma:
	and_dd_$	@ONE,@DMA_STATUS				// wait for channel 0 - could be running due to buffer refill
	mv_wi16_z	skippc,wait_prev_dma

	// setup DMA descriptor for first transfer of bitstream data
	// note: to prevent large transfer blocks, only load 1/4 buffer per single DMA transfer
	mv_dd		dma_descr.dma0.ext_byteaddr,@r0
	mv_di25		dma_descr.dma0.int_bitaddr,#bitstream_inbuf.buf
	mv_di25		dma_descr.dma0.bytelength,#BITSTREAM_INBUF_SIZE/8/4
	mv_dd		dma_descr.dma0.flags,@ZERO

	// start first DMA transfer
	mv_di25		@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|0

	// setup cyclic buffer for bitstream
	mv_wi16		aBitstreamR,#64					// prepare bitstream buffer reload when DMA has finished
	mv_ww		aBitstreamR.INCDEC,ONE
	mv_di25		@aBitstreamR.CBBASE,#bitstream_inbuf.buf
	mv_di25		@aBitstreamR.CBRANGE,#floor_log2(BITSTREAM_INBUF_SIZE)-1
	mv_di25		@aBitstreamR.CBSUBRANGE,#floor_log2(BITSTREAM_INBUF_SIZE/4)-1
	mv_di25		@aBitstreamR.CBBUFEND,#BITSTREAM_INBUF_SIZE/4*3-64

	mv_wi16		r20,#4						// buffer quarter count
quarter_buf_loop:
	// when last quarter was started: return to caller
	sub_ww_$	r20,ONE
	add_wwi32_z	pc,lr,#4
	 mv_ww_z	aBitstreamR,ZERO				// trigger bitstream buffer reload
	 mv_di25_z	@aBitstreamR.CBIRQVEC,#bitstream_read_cbam_isr.start
	 mv_wi16_z	CBSTATUS,#0					// reset cyclic buffer interrupt (for all registers)
	 or_wwi32_z	IRQ,IRQ,#IRQ_ENABLE|IRQ_ENABLE_CBAM_BCHANGE	// enable CBAM buffer change interrupts

	// otherwise wait and start next quarter
wait_curr_dma:
	and_dd_$	@ONE,@DMA_STATUS				// wait for channel 0
	mv_wi16_z	skippc,wait_curr_dma

	mv_wi16		pc,#quarter_buf_loop				// if not last quarter, proceed with next buffer quarter
	 add_di25	dma_descr.dma0.ext_byteaddr,#BITSTREAM_INBUF_SIZE/8/4
	 add_di25	dma_descr.dma0.int_bitaddr,#BITSTREAM_INBUF_SIZE/4
	 // start next DMA transfer
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|0
	 // enable stream and variable length mode on bitstream register
	 or_wwi32	MEMCFG1,MEMCFG1,#{{MCFG_STREAM|MCFG_VBLEN} << {{indexof(aBitstreamR)-8}*MCFG_SLICE_WIDTH}}
.endsection


.csection bitstream_read_cbam_isr
	.scheduling off
start:
	// 4 cycles to avoid conflicts after explicit assignment of sp (for vantage, 3 for newer SP2)
	// 3 cycles to avoid memory port ressource conflicts
	// (2 of these cycles are hidden delay slots)
	nop

	// save	working	registers to stack
	mv_wi16		CBSTATUS,#~{1<<indexof(aBitstreamR)}	// reset cyclic buffer interrupt (only for this register)
	mv_ww		(-sp+),r0
	mv_ww		(-sp+),STATUS

check_refill_next_quarter:
	and_wwi32	r0,aBitstreamR,#BITSTREAM_INBUF_SIZE-BITSTREAM_INBUF_SIZE/4	// r0 = buffer quarter of read pointer
	sub_dd		@r0,dma_descr.dma0.int_bitaddr
	sub_ww		r0,ONE
	and_wwi32_$	r0,r0,#BITSTREAM_INBUF_SIZE-BITSTREAM_INBUF_SIZE/4	// r0 = number of buffer quarters to refill * quarter buffer size
	mv_wi16_z	skippc,#exit						// return if nothing to do

wait_prev_dma:
	and_dd_$	@ONE,@DMA_STATUS					// wait for last transfer
	mv_wi16_z	skippc,wait_prev_dma

	// advance to next buffer quarter
	add_di25	dma_descr.dma0.ext_byteaddr,#BITSTREAM_INBUF_SIZE/8/4
	add_wi16_$	r0,#-BITSTREAM_INBUF_SIZE/4
	add_di25	dma_descr.dma0.int_bitaddr,#BITSTREAM_INBUF_SIZE/4
	mv_wi16_gs	pc,#check_refill_next_quarter
	 and_di25	dma_descr.dma0.int_bitaddr,#BITSTREAM_INBUF_SIZE-BITSTREAM_INBUF_SIZE/4
	 add_di25	dma_descr.dma0.int_bitaddr,#bitstream_inbuf.buf
	 // start DMA transfer
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|0
	 add_di25	@aBitstreamR.CBBUFEND,#BITSTREAM_INBUF_SIZE/4

exit:
	// restore working registers from stack and return
	mv_ww		pc,irqpc						// return from ISR
	 mv_ww		STATUS,(-sp+)
	 mv_ww		r0,(-sp+)
	 nop
	 nop
.endsection


.dsection bitstream_inbuf
  .align 0x10w			// minimum alignment required by HW
  .align BITSTREAM_INBUF_SIZE	// alignment required to simplify address calculations in ISR
  .alloc(32bit)	buf[BITSTREAM_INBUF_SIZE/32]
.endsection
