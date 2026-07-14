/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2023 videantis GmbH
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
 * FILENAME: vid_vsp_boot_loader.asm
 *
 * DESCRIPTION: videantis v-SP boot loader
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

.assume-cr CR1 = Z
.assume-cr CR2 = NZ
.assume-cr CR3 = Y
.assume-cr CR4 = NY
.assume-cr CR5 = LS
.assume-cr CR6 = GES
.assume-cr CR7 = N

.if {STACK_SIZE_IN_BITS != 0 }
.equ	STACK_SIZE	= STACK_SIZE_IN_BITS
.else
.equ	STACK_SIZE	= 1024w
.endif

.if !defined(VID_VSP_NUM_DMA_CHANNELS)
.equ VID_VSP_NUM_DMA_CHANNELS = 16
.endif

.if !defined(VID_VSP_DMA_BASE)
.equ VID_VSP_DMA_DESCR_BASE = {DMEM_BASE+DMEM_SIZE-STACK_SIZE-VID_VSP_NUM_DMA_CHANNELS*4w}
.else
.equ VID_VSP_DMA_DESCR_BASE = VID_VSP_DMA_BASE
.endif



////////////////////////////////////////////////////////////////////////////
// code section
////////////////////////////////////////////////////////////////////////////

.csection initcode

.equ VSP_WAIT_NONDMA = {1 << 16}
.equ VSP_WAIT_ALLDMA = 0xffff
.equ VSP_LRM_BASE	 = 0xC00000
.equ VSP_IRQVEC_BASE = 0xC000E0
.equ VSP_BIT_OFFSET	 =       32

.equ VSP_ADDR_GPDATA0	 = {{VSP_LRM_BASE + 0x140} * VSP_BIT_OFFSET}
.equ VSP_ADDR_DCI_IRQOUT = {{VSP_LRM_BASE + 0x14f} * VSP_BIT_OFFSET}

	.scheduling off
	.org	0w
	.export	_exit

start:
	add_$	r0,zero,one	// may be needed for GTL sim - must be first instruction

	mv_di25		@dma_descr_base,#dma_descr.start

	mv_di25		dma_descr.dma1.ext_byteaddr,#IMEM_SIZE/8	// dmem image offset
	add_dd		dma_descr.dma1.ext_byteaddr,@BOOT_ADDRESS
	mv_di25		dma_descr.dma1.int_bitaddr,#DMEM_BASE		// dmem base address
	mv_di25		dma_descr.dma1.bytelength,#DMEM_SIZE/8-VID_VSP_NUM_DMA_CHANNELS*16-STACK_SIZE/8	 // dmem size (excluding stack and DMA descriptor table)
	mv_di25		dma_descr.dma1.flags,#0

	mv_dd		dma_descr.dma0.ext_byteaddr,@BOOT_ADDRESS
	add_di25	dma_descr.dma0.ext_byteaddr,#48*4		// add offset to prevent overwriting boot code
	mv_oi32		r0,#IMEM_BASE+IMEM_BOOTCODE_SIZE
	mv_dd		dma_descr.dma0.int_bitaddr,@r0
	mv_di25		dma_descr.dma0.bytelength,#IMEM_SIZE/8-IMEM_BOOTCODE_SIZE/8
	mv_di25		dma_descr.dma0.flags,#0

	mv_di25		@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|0		// channel 0
	mv_di25		@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|1		// channel 1

	// default configuration for address registers:
	// - VBLEN mode active for all registers
	// - stack mode active for sp only
	// - positive INCDEC value for all registers (default: 1w, may be modified without restoring but must remain positive)
	// - CBAM disabled

	// initialize address registers - note: interleaved with other tasks to save NOPs
	mv_oi32		MEMCFG0,#0x88888888
	mv_oi32		MEMCFG1,#0x88888888|{MCFG_STACK<<{{indexof(sp)&7}*4}}	// enable stack mode for sp

wait_boot_dma:
	mv_dd		@r0,@DMA_STATUS
	adds_cc_$	one,r0[0:2]		// check if "finished" bits for channels 0 and 1 are set
	mv_wi16_nz	skippc,wait_boot_dma	// prev. add results in zero when DMA is done

	// manually set up a0
	mv_wi16		a0.incdec,#1w
	mv_dd		@a0.cbbase,@zero
	mv_dd		@a0.cbrange,@mone
	mv_dd		@a0.cbsubrange,@mone
	mv_oi32		a0,#@a1.incdec		// for init_aregs loop

	// loop over remaining address registers
	mv_wi16		r0,#15
init_aregs:
	// sp2.0 fix: hardware loads boot loader using an arbitrary channel without auto_clear; reset channels 1..15 to "free" here,
	//              channel 0 (and also 1) has been cleared using CLEAR_AUTO after boot DMA
	or_wwi32	r1,r0,#DC_CLEAR
	mv_dd		@DMA_CTRL,@r1
	eloop_wi14	r0,init_aregs
	 mv_oi32	(a0)[@a1.incdec-@a1.incdec],#1w
	 mv_oo		(a0)[@a1.cbbase-@a1.incdec],zero
	 mv_oo		(a0)[@a1.cbrange-@a1.incdec],mone
	 mv_oo		(-a0+)[@a1.cbsubrange-@a1.incdec],mone

	// prepare init_regs loop
	mv_oi32		a1,#@a2
	mv_oi32		a0,#@r3
	mv_wi16		r2,#7			// reset a2..a15, r3..r30: 14 aregs, 28 regs = 7 iterations

	// setup stack growing upward
	mv_wi16		sp.incdec,#-1w

init_regs:
	.warning off,datalatency
	mv_wi16		(-a1+),#0		// at least 3 - not 4 - independent instructions required after a1 setup
	.warning on
	mv_wi16		(-a1+),#0
	eloop_wi14	r2,init_regs
	 mv_wi16	(-a0+),#0
	 mv_wi16	(-a0+),#0
	 mv_wi16	(-a0+),#0
	 mv_wi16	(-a0+),#0

	mv_oi32		sp,#stack

	mv_dd	@r0, SP_VERSION
	lsr_wwi5 r0, r0, #VIDASM_PID2
	sub_wi32w_$ r0, #VIDASM_PID1, r0
	mv_di25_nz GPDATA7, #0x0f
	mv_di25_nz @pc, #endless

	// call main routine with arguments set
	mv		pc,#_main		// do not force mv_wi16 here to enable simulation with large IMEM
	 mv_oi32	condsel,#VIDASM_SP2_CONDSEL
	 mv_dd		@r0,#0x7fde00w		// argc for vid_sim semihosting (not chip-it compatible)
	 mv_oi32	r1,#0x7fde01w		// argv[]
	 mv_wi16	lr,#_exit-4		// let main return to _exit (if it returns)

_exit:
	wait #VSP_WAIT_ALLDMA    // check if all 16 DMA channels are ready
	 mv VSP_ADDR_GPDATA0, R0 // write exit(int) argument to GPDATA0
	 nop
	 nop
	 nop

endless:
	wait #VSP_WAIT_NONDMA //endless WAIT
	 nop
	 nop
	 nop
	 nop
.endsection


////////////////////////////////////////////////////////////////////////////
// data section
////////////////////////////////////////////////////////////////////////////

.dsection stack
	.org	DMEM_BASE+DMEM_SIZE-STACK_SIZE
	.align  1w
start:
.export stack
	.alloc(STACK_SIZE)	stack
.endsection


.dsection dma_descr
    .org   {VID_VSP_DMA_DESCR_BASE}
    .align 64w
start:
    .loop    dma=[0..VID_VSP_NUM_DMA_CHANNELS-1]
start:
      .alloc(1w)    int_bitaddr
      .alloc(1w)    ext_byteaddr
      .alloc(1w)    bytelength
      .alloc(1w)    flags
    .endloop

.endsection

