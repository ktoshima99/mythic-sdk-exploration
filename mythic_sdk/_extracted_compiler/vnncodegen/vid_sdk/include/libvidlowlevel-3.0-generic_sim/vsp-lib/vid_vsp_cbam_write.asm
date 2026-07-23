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
* FILENAME: vid_vsp_cbam_write.asm
*
* DESCRIPTION: bitstream cyclic buffer write functions
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/


.equ FULL_BUFFER_ALERT		= 0
.equ IMMEDIATE_SYNC 		= 0
.equ BITSTREAM_BUF_SIZE		= 0x80w
.equ NEW_CBAM			= 1

.if {!defined(SCEP_HOOK)}
.equ SCEP_HOOK			= 0
.endif


// function _vid_bitstream_init_write
// arguments:
//  r0:	idx
//  r1: BITSTREAM_BASE
// return value: none
.csection vid_bitstream_init_write
	.scheduling off
	.export	_vid_bitstream_init_write

_vid_bitstream_init_write:
	and_wwi32	IRQ,IRQ,#~IRQ_ENABLE_CBAM_BCHANGE		// disable CBAM interrupt
	mv		(-sp+), at0					// save working registers
 	mv		(-sp+), at1
 	mv		(-sp+), r2
.if {1 == SCEP_HOOK}
	mv		(-sp+), lr
.endif

.if { FULL_BUFFER_ALERT == 1 }
	lsl_wwi5	at1, r0, #5  					// let at1 = &buffer_limits[r0]
	add_wwi32	at1, at1, #bitstream_data.buffer_limits		// array element size: 32bit
.endif

	lsl_wwi5	at0, r0, #5 					// let at0 = &areg_buffer[r0]
	add_wwi32	at0, at0, #bitstream_data.areg_buf		// array element size: 32bit

	lsl_wwi5	at1, r0, #7   					// let at1 = &dma_descr[r0]
	add_wwi32	at1, at1, #dma_descr.start			// array element size = 4*32bit

.if { FULL_BUFFER_ALERT == 1 }
	mv_oo		(at1), mone	// WAVEPIPELINING! set buffer limit to 0xffffffff (= no limit)
.endif

	or_www_$	zero, zero, r0					// skip cbam init part for idx > 0
	mv_wi16_nz	skippc, #init_state

	mv_dd		cut_off_pad_bytes, @zero
	mv_ww		aBitstreamW, zero				// setup cyclic buffer for bitstream
	mv_ww		aBitstreamW.INCDEC, one
	mv_di25		@aBitstreamW.CBBASE, #bitstream_data.buf
	mv_di25		@aBitstreamW.CBRANGE, #floor_log2(BITSTREAM_BUF_SIZE)-1
	mv_di25		@aBitstreamW.CBSUBRANGE, #floor_log2(BITSTREAM_BUF_SIZE/4)-1
	mv_di25		@aBitstreamW.CBIRQVEC, #bitstream_cbam_isr.start

	.equ 		MEMCFG_MASK = {MCFG_VBLEN << {{indexof(aBitstreamW)-8}*MCFG_SLICE_WIDTH}}
	or_wwi32	MEMCFG1, MEMCFG1, #MEMCFG_MASK			// enable vblen mode on bitstream areg


	mv_oi32		CBSTATUS,#~{1<<indexof(aBitstreamW)}		// reset cyclic buffer interrupt
	mv_dd		bitstream_data.cur_state, @zero 		// init to state 0

	mv_di25		@aBitstreamW.CBBUFEND, #{{BITSTREAM_BUF_SIZE/4}*2}-1	// set cbbufend limit

init_state:
	or_wwi32	IRQ,IRQ,#IRQ_ENABLE|IRQ_ENABLE_CBAM_BCHANGE	// enable CBAM buffer change interrupts

wait_prev_dma:							// wait until DMA channel <cur_state> is available
	mv_dd		@r2, bitstream_data.cur_state
	lsl_www		r2, one, r2
	and_dd_$	@r2, @DMA_STATUS
	mv_wi16_z	skippc, #wait_prev_dma

	mulus_wwi32	(at1), r0, #BITSTREAM_BUF_SIZE			// set initial int_bitaddr
	add_wi16	(at1), #bitstream_data.buf

.if {1 == SCEP_HOOK}
	mv		lr, #scep_init_exit
        mv_wi16		skippc, #scep_init
.endif
scep_init_exit:

	// setup DMA descriptor for first transfer of bitstream data
	mv_oo		(at1)[1w], r1		// ext_byteaddr
	mv_oo		(at1)[2w], zero		// bytelength
	mv_oo		(at1)[3w], zero		// flags

	mv_oo		(at0), zero	        // initial areg

.if {1 == SCEP_HOOK}
	 mv		lr, (-sp+)
.endif
	mv		r2, (-sp+)		// restore working registers
	mv		at1, (-sp+)
	mv		at0, (-sp+)

	add_wwi9	pc,lr,#4		// return to caller
	 nop
	 nop
	 nop
	 nop
.endsection




/*
// function _vid_bitstream_flush
// arguments: none
// return value: none
.csection bitstream_flush
	.scheduling off
	.export	_vid_bitstream_flush
_vid_bitstream_flush:

	// save working registers on stack
	mv_ww		(-sp+),r0
	mv_ww		(-sp+),r1
	mv_ww		(-sp+),at0

.if { SCEP_HOOK == 1 }
	mv_wi16		SKIPPC, #scep_hook_flush
.endif
scep_hook_flush_exit:

	mv_dd		@r0, bitstream_data.cur_state
	lsl_wwi5	at0, r0, #7   // 4*32bit => *128 => <<7
	add_wwi32	at0, at0, #dma_descr.start

	// wait until DMA channel <cur_state> is available
wait_prev_dma:
	lsl_www		r1, one, r0
	and_dd_$	@r1, @DMA_STATUS
	mv_wi16_z	skippc, #wait_prev_dma

.if {1} // was: {1 == USE_SP3_CVU}
	mv_dd		@r0, @aBitstreamW.CBBUFEND
	add_wwi32	r0, r0, #{BITSTREAM_BUF_SIZE/4}
	mv_dd		@aBitstreamW.CBBUFEND, @r0
.endif

	// calculate new external transfer address
	mv_oo		r0, (at0)[2w]				// r0 = dma.bytelength
	add_oo		(at0)[1w], r0				// dma.ext_byteaddr += dma.bytelength

.if {defined (NEWCBAM)}
	add_wwi32	r0, aBitstreamW, #-BITSTREAM_BUF_SIZE/4			 // set r0 to ..
	and_wwi32	r0, r0, #{BITSTREAM_BUF_SIZE-1}-{BITSTREAM_BUF_SIZE/4-1} // ..start of prev. subbuffer
	add_dd		@r0, @aBitstreamW.CBBASE				 // add base offset
	mv		(at0), r0						 // internal address
.else
	lsl_wwi5	r0, r0, #3				// r0 = last bits transferred
	add_www		(at0), (at0), r0			// dma.int_bitaddr += r0*8

	// internal bitstream data buffer wrap around
	mv_dd		@r0, @aBitstreamW.CBBASE
	sub_www		(at0), (at0), r0
	and_wwi32	(at0), (at0), #BITSTREAM_BUF_SIZE-1
 	add_www		(at0), (at0), r0
.endif

// VANTAGE WORKAROUND
security_wait_dma:
	mv_dd		@r1, @DMA_STATUS
	or_wwi32	r1, r1, #0xffff0000 // ignore upper 16 bits
	sub_www_$	zero, r1, mone
	mv_wi16_nz	skippc, #security_wait_dma
	 nop
	 nop
	 nop
	 nop
// VANTAGE WORKAROUND ENDS

	// get number of bytes in current sub-buffer
	// dma.bytelength = ((aBitstreamW & SUBRANGE_MASK)+7)/8
	and_wwi32	r0, aBitstreamW, #{BITSTREAM_BUF_SIZE/4}-1
	add_wwi9	r0, r0, #7
	lsr_wwi5	r0, r0, #3
	mv_oo_$		(at0)[2w], r0

	// dma.flags      = zero
	mv_dd		dma_descr.dma0.flags, @zero

.if defined(USE_COPYMACHINE) && {USE_COPYMACHINE == 1}
	mv	(-sp+), lr
	trp_wwi32_nz	lr, pc, #copymachine_handler
	 nop
	 nop
	 nop
	 nop
	mv	lr, (-sp+)

.else
	// start DMA transfer, if bytelength nonzero
	mv_oi32		r0, #{DC_CLEAR_AUTO|DC_DIR_OUT}
	or_dd		@r0, bitstream_data.cur_state
	mv_dd_nz	@DMA_CTRL, @r0
.endif

	// always wait for end oft transfer
	mv_dd		@r0, bitstream_data.cur_state
	nop
	nop
	nop
wait_dma:
	lsl_www		r1, one, r0
	and_dd_$	@r1, @DMA_STATUS
	mv_wi16_z	skippc, #wait_dma
	 nop
	 nop
	 nop
	 nop

exit:
	// return to caller
	mv_ww		at0, (-sp+)
	add_wwi9	pc,lr,#4
	 // restore working registers from stack
	 mv_ww		r1, (-sp+)
	 mv_ww		r0, (-sp+)
	 nop
	 nop
.endsection
*/


// function _vid_bitstream_get_length
// arguments:
//  r0: BITSTREAM_BASE
// return value:
//  r0: length of bitstream in bytes
.csection bitstream_get_length
	.scheduling off
	.export	_vid_bitstream_get_length
_vid_bitstream_get_length:
	// save working regs
	mv_ww		(-sp+), at1

	// make at1 point to current dma
	mv_dd		@at1, bitstream_data.cur_state
	lsl_wwi5	at1, at1, #7   // 4*32bit => *128 => <<7
	add_wwi32	at1, at1, dma_descr.start

	nop
	nop
	nop

	sub_oo		r0, (at1)[2w]		// r0 = base - bytelength
	sub_oo		r0, (at1)[1w]		// r0 = base - bytelength - ext_addr

	add_wwi9	pc,lr,#4
	// restore at1, still using "old" value in the following delay slots
	 mv_ww		at1, (-sp+)
	 sub_www	r0, zero, r0		// r0 = (ext_addr+bytelength) - base
	 nop
	 nop

.endsection




// function _vid_bitstream_set_buffer
// arguments:
//  r0: bitstream index
//  r1: bitstream buffer address
//  r2: bitstream buffer limit
// return value: none
.csection bitstream_set_buffer
	.scheduling off
	.export _vid_bitstream_set_buffer
_vid_bitstream_set_buffer:

	// save at0, let at0 = &dma_descr[r0]
	mv		(-sp+), at0
	lsl_wwi5	at0, r0, #7                 // 4*32bit => *128 => <<7
	add_wwi32	at0, at0, #dma_descr.start

	// save at1, let at1 = &buffer_limits[r0]
 	mv		(-sp+), at1
	lsl_wwi5	at1, r0, #5                 // 1*32bit => *32 => <<5
	add_wwi32	at1, at1, #bitstream_data.buffer_limits

	// set dma_descr[r0].ext_byte_addr = r1
	mv_oo		(at0)[1w], r1

	// return
	add_wwi9	pc,lr,#4

 	 mv		at1, (-sp+)
	 // WAVE-PIPELINING (at1)!
	 // set buffer_limits[idx] = r2
  	 mv_oo		(at1)[1w], r2
 	 mv		at0, (-sp+)
	 nop
.endsection



.csection bitstream_cbam_isr
	.scheduling off
start:
	// 4 cycles to avoid conflicts after exclusive assignment of sp
	// 3 cycles to avoid memory port ressource conflicts
	// (2 of these cycles are delay slots)
	nop
	// reset cyclic	buffer interrupt
	mv_oi32		CBSTATUS, #~{1<<indexof(aBitstreamW)}

	// save	working	registers to stack
	mv_ww		(-sp+), r0
	mv_ww		(-sp+), r1
	mv_ww		(-sp+), STATUS
	mv_ww		(-sp+), at0


	mv_dd		@r0, bitstream_data.cur_state
	lsl_wwi5	at0, r0, #7   // 4*32bit => *128 => <<7
	add_wwi32	at0, at0, #dma_descr.start

.if { FULL_BUFFER_ALERT == 1 }
	// make at1 point to current bitstream's entry in ext_buffer_bases array
	trp_www		(-sp+), at1, r0
	lsl_wwi5	at1, at1, #5
	add_wwi32	at1, at1, #bitstream_data.buffer_limits
.endif

wait_prev_dma:
	lsl_www		r1, one, r0
	and_dd_$	@r1, @DMA_STATUS			// wait for channel "cur_state"
	mv_wi16_z	skippc,wait_prev_dma


.if {1} // was: {1 == USE_SP3_CVU}
	mv_dd		@r0, @aBitstreamW.CBBUFEND
	add_wwi32	r0, r0, #{BITSTREAM_BUF_SIZE/4}
	mv_dd		@aBitstreamW.CBBUFEND, @r0
.endif

	// calculate new external transfer address
	mv_oo		r0, (at0)[2w]				// r0 = dma.bytelength
	add_oo		(at0)[1w], r0				// dma.ext_byteaddr += r0

.if { FULL_BUFFER_ALERT == 1}
	// if(dma.ext_byteaddr >= buffer_limit) alert();
	mv_oo		r1, (at0)[1w]
	sub_ww_$	r1, (at1)
	mv_wi16_geu	skippc, #bs_full_buffer_alert
.endif
full_buffer_alert_exit:


	// calculate new internal transfer address
.if {defined(NEW_CBAM)}
	add_wwi32	r0, aBitstreamW, #-BITSTREAM_BUF_SIZE/4			 // set r0 to ..
	and_wwi32	r0, r0, #{BITSTREAM_BUF_SIZE-1}-{BITSTREAM_BUF_SIZE/4-1} // ..start of prev. subbuffer
	add_dd		@r0, @aBitstreamW.CBBASE				 // add base offset
	mv		(at0), r0						 // internal address
.else
	lsl_wwi5	r0, r0, #3				// r0 = last bits transferred
	add_www		(at0), (at0), r0			// dma.int_bitaddr += r0*8

	// internal bitstream data buffer wrap around
	mv_dd		@r0, @aBitstreamW.CBBASE
	sub_www		(at0), (at0), r0
	and_wwi32	(at0), (at0), #BITSTREAM_BUF_SIZE-1
 	add_www		(at0), (at0), r0
.endif

	// misc. settings for new transfer
	mv_oi32		(at0)[2w], #{BITSTREAM_BUF_SIZE >> 3}/4	// dma.bytelength = sizeof(SUB_BUFFER)
	mv_oo		(at0)[3w], zero				// dma.flags      = 0

.if { SCEP_HOOK == 1 }
	// input: r0: address of current subbuffer to transfer
	mv_ww		r0, (at0)
	mv_wi16		SKIPPC, #scep_hook_isr
	nop
	nop
	nop
	nop
.endif
scep_hook_isr_exit:

	mv_oo		r0, (at0)[2w]
	sub_dd		@r0, cut_off_pad_bytes			// dma.bytelength  -= cut_off
	mv_oo		(at0)[2w], r0
	mv_dd		cut_off_pad_bytes, @zero

.if defined(USE_COPYMACHINE) && {USE_COPYMACHINE == 1}
	mv		(-sp+), lr
	trp_wwi32	lr, pc, #copymachine_handler
	 nop
	 nop
	 nop
	 nop
	mv		lr, (-sp+)

.else
	// start DMA transfer
	mv_dd		@r0, bitstream_data.cur_state
	or_wwi32	r0, r0, #DC_CLEAR_AUTO|DC_DIR_OUT
	mv_dd		@DMA_CTRL, @r0
.endif

.if { IMMEDIATE_SYNC !=	0 }
	nop
	nop
	nop
	nop
wait_ch0:
	lsl_www		r1, one, r0
	and_dd_$	@r1,@DMA_STATUS			// check if curr. memcpy on channel 0 is complete
	mv_wi16_z	skippc,#wait_ch0
	 nop
	 nop
	 nop
	 nop
.endif

exit:
.if { FULL_BUFFER_ALERT == 1}
	mv_ww		at1, (-sp+)
.endif
	mv_ww		pc,irqpc				// return from ISR
	 // restore working registers from stack
 	 mv_ww		at0, (-sp+)
 	 mv_ww		STATUS,(-sp+)
	 mv_ww		r1,(-sp+)
	 mv_ww		r0,(-sp+)

bs_full_buffer_alert:
.if { FULL_BUFFER_ALERT == 1 }

	// prepare to enter C-function, save a lot of registers
	mv_ww		(-sp+), lr
	mv_ww		(-sp+), r0
	mv_ww		(-sp+), r1

	// arguments for C-function:
	//  r0:	index of current bitstream
	//  r1:	current buffer address (exceeding the limit and
	//	therefore triggering the C-function call)
	mv_dd		@r0, bitstream_data.cur_state
	mv_ww		r1, (at1)

	mv_ww		(-sp+), r20
	mv_ww		(-sp+), r21
	mv_ww		(-sp+), r22
	mv_ww		(-sp+), r23
	trp_wwi32	lr, pc, #_bs_full_buffer_handler
	 mv_ww		(-sp+), r24
	 mv_ww		(-sp+), r25
	 mv_ww		(-sp+), r26
	 mv_ww		(-sp+), r27

	// evaluate return value
	or_www_$	zero, r0,r0

	mv_ww		r27, (-sp+)
	mv_ww		r26, (-sp+)
	mv_ww		r25, (-sp+)
	mv_ww		r24, (-sp+)
	mv_ww		r23, (-sp+)
	mv_ww		r22, (-sp+)
	mv_ww		r21, (-sp+)
	mv_ww		r20, (-sp+)
	mv_wi16		pc, #full_buffer_alert_exit
	 mv_wi16_z	pc, #exit
	 mv_ww		r1, (-sp+)
	 mv_ww		r0, (-sp+)
	 mv_ww		lr, (-sp+)
.endif

.endsection

.csection bitstream_flush
	.scheduling off
	.export	_vid_bitstream_flush
_vid_bitstream_flush:
	.equ		r_tmp	       = r0
	.equ		r_pad_count    = r1
	.equ		r_pad_left     = r24

	mv_ww		(-sp+), r_tmp
	mv_ww		(-sp+), r_pad_count
	mv_ww		(-sp+), r_pad_left

// VANTAGE WORKAROUND
security_wait_dma:
	mv_dd		@r_tmp, @DMA_STATUS
	or_wwi32	r_tmp, r_tmp, #0xffff0000 // ignore upper 16 bits
	sub_www_$	zero, r_tmp, mone
	mv_wi16_nz	skippc, #security_wait_dma
	 nop
	 nop
	 nop
	 nop
// VANTAGE WORKAROUND ENDS

	and_$		r_tmp, aBitstreamW, #BITSTREAM_BUF_SIZE/4-1 // bit offset within current subbuffer
	mv_z		skippc, #exit

	sub		r_tmp, #BITSTREAM_BUF_SIZE/4, r_tmp	// # of bits left in current subbuffer
	lsr		r_pad_count, r_tmp, #3			// # of pad bytes ..
	mv_dd		cut_off_pad_bytes, @r_pad_count		// .. to be cut off

pad_128:
	lsr_$		r_pad_count,  r_tmp, #7			// bits left to pad / 128
	and		r_pad_left, r_tmp, #0x7f		// bits left to pad % 128
	mv_z		skippc, #pad_32
	 nop
	 nop
	 nop

pad_128_loop:
	 eloop		r_pad_count, pad_128_loop
	  mvu_cc	(-aBitstreamW+)[0:32], mone[0:32]	// pad 32bit word
	  mvu_cc	(-aBitstreamW+)[0:32], mone[0:32]	// pad 32bit word
	  mvu_cc	(-aBitstreamW+)[0:32], mone[0:32]	// pad 32bit word
	  mvu_cc	(-aBitstreamW+)[0:32], mone[0:32]	// pad 32bit word

pad_32:
	lsr_$		r_pad_count,  r_pad_left, #5      	// bits left to pad / 32
	and		r_pad_left, r_pad_left, #0x1f		// bits left to pad % 32
	mv_z		skippc, #pad_bits
	 nop
	 nop
	 nop

pad_32_loop:
	eloop		r_pad_count, #pad_32_loop
	 mvu_cc		(-aBitstreamW+)[0:32], mone[0:32]	// pad 32bit word
	 nop
	 nop
	 nop

pad_bits:
	mv_vw		(-aBitstreamW+)[0:r_pad_left], mone	// pad remainder
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
scep_hook_flush_exit:
wait_dma:							// wait until DMA channel <cur_state> is available
	mv_dd		@r_tmp, bitstream_data.cur_state
	lsl_www		r_tmp, one, r_tmp
	and_dd_$	@r_tmp, @DMA_STATUS
	mv_wi16_z	skippc, #wait_dma
	 nop
	 nop
	 nop
	 nop
exit:

	add_wwi9	pc,lr,#4
	 mv_ww		r_pad_left,	(-sp+)
	 mv_ww		r_pad_count,	(-sp+)
 	 mv_ww		r_tmp, 		(-sp+)
	 nop
.endsection

// arguments:
//  r0: index
.csection bitstream_switch
	.scheduling off
	.export _vid_bitstream_switch
_vid_bitstream_switch:

	// save working registers to stack
	mv_ww		(-sp+), at0

	// get pointer in areg_buf to store current areg
	mv_dd		@at0, bitstream_data.cur_state
	lsl_wwi5	at0, at0, #5     // *1*32bit => *32 => <<5
	add_wwi32	at0, at0, #bitstream_data.areg_buf

	// prevent cbam interrupt when changing aBitstream
	mv_di25		@aBitstreamW.CBSUBRANGE, #-1

	 // set new state
	mv_dd		bitstream_data.cur_state, @r0

	nop

	// save old state
	mv_ww		(at0), aBitstreamW

	// get pointer to new areg_buf
	lsl_wwi5	at0, r0, #5     // *1*32bit => *32 => <<5
	add_wwi32	at0, at0, #bitstream_data.areg_buf

	 // set new cbrange
	mulus_wwi32	r0, r0, #BITSTREAM_BUF_SIZE
	add_wi16	r0, #bitstream_data.buf
	mv_dd		@aBitstreamW.CBBASE, @r0

	// load new state
	mv_ww		aBitstreamW, (at0)

.if {1} // was: {1 == USE_SP3_CVU}
	// set CBBUFEND register to end of buffer
	and_wwi32	r0, aBitstreamW, {~{{BITSTREAM_BUF_SIZE/4}-1}}
	add_wwi32	r0, r0, #{{BITSTREAM_BUF_SIZE/4}*2}-1
	mv_dd		@aBitstreamW.CBBUFEND, @r0
.endif

	mv_di25		@aBitstreamW.CBSUBRANGE,#floor_log2(BITSTREAM_BUF_SIZE/4)-1
	mv_oi32		CBSTATUS,#~{1<<indexof(aBitstreamW)}		// reset cyclic buffer interrupt

	// restore working registers from stack
 	add_wwi9	pc,lr,#4
	 mv_ww		at0, (-sp+)
	 nop
	 nop
	 nop

.endsection


.dsection bitstream_data
  .align BITSTREAM_BUF_SIZE
  .alloc (1w) buf[MP_CORES*BITSTREAM_BUF_SIZE/32]

  .align 1w
  .alloc (1w) cur_state

  .alloc (1w) cut_off_pad_bytes
  .export cut_off_pad_bytes

// array to hold current adress register state for each bitstream
areg_buf:
  .loop state=[0..MP_CORES-1]
    .alloc (1w) areg
  .endloop

// array to hold buffer limits
_buffer_limits:
buffer_limits:
.if { FULL_BUFFER_ALERT == 1 }
  .loop state1=[0..MP_CORES-1]
    .alloc(1w) limit
  .endloop
.endif

.export _buffer_limits

.endsection
