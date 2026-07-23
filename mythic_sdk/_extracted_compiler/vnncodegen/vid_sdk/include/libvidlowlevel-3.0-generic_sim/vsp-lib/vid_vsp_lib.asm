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
* FILENAME: vid_vsp_lib.asm
*
* DESCRIPTION: v-SP library functions
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// pre-allocated:
// a13: stack pointer (sp)
// r31: link register (lr)

// note: C compiler requires every register to be restored upon return to calling C code

.equ at0		= a0		// local temp registers
.equ at1		= a1

.equ aBitstreamR	= a14
.equ aBitstreamW        = a15

//.equ rt0		= r0		// local temp registers
//.equ rt1		= r1
//.equ rt2		= r2
//.equ rt3		= r3

.equ rv0		= r24		// local temp register for variable-length parameters


/* ATTENTION: following section only valid for CABAC code */
//
//// globally defined registers
//

// bitstream pointer
.equ Abitstream = A14

// stack pointer
.equ Astack	= A13

// variable bitlength register
.equ Rvlen	= R30

// return address for subroutine calls and value returned by main
.equ Rreturn	= R31

// define temporary general purpose registers
.equ Rtmp0	= R0
.equ Rtmp1	= R1
.equ Rtmp2	= R2
.equ Rtmp3	= R3

// define temporary address registers
.equ Atmp0	= A0
.equ Atmp1	= A1
.equ Atmp2	= A2
.equ Atmp3	= A3

// bitlength register for WV and VW operations
.equ Rbitlen    = R24

// usercode word start address (must be 64-bit aligned for DMA transfer)
.equ USERCODE_START         = 0x00030

// dmem size and number of banks (1 bank x 8192 word x 32 bit = 1 x 32 kbyte)
.equ SP2_DMEMBANK_ADDRWIDTH = 13                                          // number of address bits for dmem
.equ SP2_DMEMBANK_SIZE      = 1w << SP2_DMEMBANK_ADDRWIDTH                // size of single dmem bank
.equ SP2_DMEMBANK_RANGE     = 1                                           // number of dmem banks
.equ SP2_DMEM_SIZE          = SP2_DMEMBANK_RANGE * SP2_DMEMBANK_SIZE      // data memory size

// imem size and number of banks (1 bank x 4096 word x 64 bit = 1 x 32 kbyte)
.equ SP2_IMEMBANK_ADDRWIDTH = 12                                          // number of address bits for imem
.equ SP2_IMEMBANK_SIZE      = 2w << SP2_IMEMBANK_ADDRWIDTH                // size of single imem bank
.equ SP2_IMEMBANK_RANGE     = 1                                           // number of imem banks
.equ SP2_IMEM_SIZE          = SP2_IMEMBANK_RANGE * SP2_IMEMBANK_SIZE      // instruction memory size

// size and number of intercore memory (2 memories x 512 word x 32 bit = 2x 2 kbyte)
.equ SP2_ICMEM_ADDRWIDTH    = 9
.equ SP2_ICMEM_SIZE         = 1w << SP2_ICMEM_ADDRWIDTH
.equ SP2_ICMEM_RANGE        = 2
.equ SP2_ICMEM_BANK_MASK    = 0x003C0000w

// memory base addresses
.equ SP2_DMEM_BASE          = 0x00000000w                       // start of data memory
.equ SP2_DMEM_END           = SP2_DMEM_BASE + SP2_DMEM_SIZE - 1 // end of data memory

.equ SP2_IMEM_BASE          = 0x00200000w                       // start of instruction memory
.equ SP2_IMEM_END           = SP2_IMEM_BASE + SP2_IMEM_SIZE - 1 // end of instruction memory

.equ SP2_ICMEM_BASE         = 0x00400000w                       // start of first intercore memory
.equ SP2_ICMEM_OFFS         = 0x00040000w                       // offset to next intercore memory
.equ SP2_ICMEM_END          = 0x00800000w                       // end of intercore memory space

.equ SP2_REG_BASE           = 0x00C00000w                       // start of register file windows
.equ SP2_GAMSIZE            = 0x00FFFFFFw                       // size of global address map

// CONDSEL register modes
.equ SP2_COND_ALWAYS	    = 0b0000    // execute uncoditionally
.equ SP2_COND_Y		    = 0b0010	// Carry
.equ SP2_COND_LU	    = 0b0010	// Less unsigned
.equ SP2_COND_NY	    = 0b0011	// No carry
.equ SP2_COND_GEU	    = 0b0011	// Greater or equal unsigned
.equ SP2_COND_V		    = 0b0100	// Overflow
.equ SP2_COND_NV	    = 0b0101	// No overflow
.equ SP2_COND_Z		    = 0b0110	// Zero
.equ SP2_COND_E		    = 0b0110	// Equal
.equ SP2_COND_NZ	    = 0b0111	// Not zero
.equ SP2_COND_NE	    = 0b0111	// Not equal
.equ SP2_COND_N		    = 0b1000	// Negative
.equ SP2_COND_NN	    = 0b1001	// Not negative
.equ SP2_COND_P		    = 0b1001	// Positive
.equ SP2_COND_LES	    = 0b1010	// Less or equal signed
.equ SP2_COND_GS	    = 0b1011	// Greater signed
.equ SP2_COND_S		    = 0b1100	// Signed
.equ SP2_COND_LS	    = 0b1100	// Less signed
.equ SP2_COND_NS	    = 0b1101	// Not signed
.equ SP2_COND_GES	    = 0b1101	// Greater or equal signed
.equ SP2_COND_LEU	    = 0b1110	// Less or equal unsigned
.equ SP2_COND_GU	    = 0b1111	// Greater unsigned

/* ATTENTION: END of section only valid for CABAC code */

// common register definitions
.equ sp			= a13
.equ lr			= r31

// memory addresses
.equ DMEM_BASE		= 0x00000000w	// start of data memory (for DMA)
.equ DMEM_SIZE		= @DMEM_END-@DMEM_START

.equ IMEM_BASE		= 0x00200000w	// start of instruction memory (for DMA)
.equ IMEM_SIZE		= @IMEM_END-@IMEM_START
.equ IMEM_BOOTCODE_SIZE	= 0x0030w

// CONDSEL codes
.equ COND_ALWAYS	= 0b0000	// execute unconditionally
.equ COND_Y		= 0b0010	// carry
.equ COND_LU		= 0b0010	// less unsigned
.equ COND_NY		= 0b0011	// no carry
.equ COND_GEU		= 0b0011	// greater or equal unsigned
.equ COND_V		= 0b0100	// overflow
.equ COND_NV		= 0b0101	// no overflow
.equ COND_Z		= 0b0110	// zero
.equ COND_E		= 0b0110	// equal
.equ COND_NZ		= 0b0111	// not zero
.equ COND_NE		= 0b0111	// not equal
.equ COND_N		= 0b1000	// negative
.equ COND_P		= 0b1001	// positive
.equ COND_LES		= 0b1010	// less or equal signed
.equ COND_GS		= 0b1011	// greater signed
.equ COND_LS		= 0b1100	// less signed
.equ COND_GES		= 0b1101	// greater or equal signed
.equ COND_LEU		= 0b1110	// less or equal unsigned
.equ COND_GU		= 0b1111	// greater unsigned

// rounding modes
.equ SP2_RM_TRUNC_DN	= 0b00		// truncate down to nearest integer
.equ SP2_RM_TRUNC_UP	= 0b01		// truncate up to nearest integer
.equ SP2_RM_ROUND_DN	= 0b10		// round down to nearest integer
.equ SP2_RM_ROUND_UP	= 0b11		// round up to nearest integer

// memory config modes
.equ MCFG_SLICE_WIDTH	= 4		// number of memory configuration bits per register

.equ MCFG_RESERVED	= 0x01		// reserved
.equ MCFG_STACK		= 0x02		// stack mode: effect of increment/decrement is inversed for writebacks to operand d
.equ MCFG_STREAM	= 0x04		// bitstream buffer modus: access to bitstream via SBUFFER registers
.equ MCFG_VBLEN		= 0x08		// variable bitlength mode: address increment/decrement depends on operand width
.equ MCFG_ALL		= 0x0F		// all memory modes activated

// layout of interrupt register
.equ IRQ_ENABLE_BIT			= 0	// global interrupt enable
.equ IRQ_BUSY_BIT			= 1	// interrupt handler is busy
.equ IRQ_ENABLE_MEM_FAILURE_BIT		= 2	// enable memory failure interrupts
.equ IRQ_ENABLE_CBAM_BCHANGE_BIT	= 3	// enable cyclic buffer interrupts
.equ IRQ_ENABLE_TIMER_BIT		= 4
.equ IRQ_ENABLE_DMA_DONE_BIT		= 5
.equ IRQ_ENABLE_CEU_BIT			= 6

.equ IRQ_ENABLE			= 1 << IRQ_ENABLE_BIT			// global interrupt enable
.equ IRQ_BUSY			= 1 << IRQ_BUSY_BIT			// interrupt handler is busy
.equ IRQ_ENABLE_MEM_FAILURE	= 1 << IRQ_ENABLE_MEM_FAILURE_BIT	// enable memory failure interrupts
.equ IRQ_ENABLE_CBAM_BCHANGE	= 1 << IRQ_ENABLE_CBAM_BCHANGE_BIT	// enable cyclic buffer interrupts
.equ IRQ_ENABLE_TIMER		= 1 << IRQ_ENABLE_TIMER_BIT
.equ IRQ_ENABLE_DMA_DONE	= 1 << IRQ_ENABLE_DMA_DONE_BIT
.equ IRQ_ENABLE_CEU		= 1 << IRQ_ENABLE_CEU_BIT

// layout of MEMSTATUS register
.equ MEMS_ACCESS_FAILURE_DMEM0	= 1 << 0
.equ MEMS_ACCESS_FAILURE_DMEM1	= 1 << 1

// layout of STATUS register
.equ STATUS_CARRY		= 0
.equ STATUS_OVERFLOW		= 1
.equ STATUS_ZERO		= 2
.equ STATUS_NEGATIVE		= 3
.equ STATUS_SIGN		= 4

.equ STATUS_CARRY_MASK		= 1 << STATUS_CARRY
.equ STATUS_OVERFLOW_MASK	= 1 << STATUS_OVERFLOW
.equ STATUS_ZERO_MASK		= 1 << STATUS_ZERO
.equ STATUS_NEGATIVE_MASK	= 1 << STATUS_NEGATIVE
.equ STATUS_SIGN_MASK		= 1 << STATUS_SIGN

// DMACTRL register
.equ DC_POS_CHID	= 0
.equ DC_POS_DIR		= 4
.equ DC_POS_AUTO_CLEAR	= 5
.equ DC_POS_IRQEN	= 6
.equ DC_POS_CONST	= 7
.equ DC_POS_TAG		= 8
.equ DC_POS_PRIO	= 12
.equ DC_POS_CLEAR	= 16
.equ DC_DIR_OUT		= {0<<DC_POS_DIR}
.equ DC_DIR_IN		= {1<<DC_POS_DIR}
.equ DC_CONST		= {1<<DC_POS_CONST}
.equ DC_CLEAR_MANU	= {0<<DC_POS_AUTO_CLEAR}
.equ DC_CLEAR_AUTO	= {1<<DC_POS_AUTO_CLEAR}
.equ DC_IRQEN		= {1<<DC_POS_IRQEN}
.equ DC_CLEAR		= {1<<DC_POS_CLEAR}

// old (SP2):
// TMRCTRL register
.equ TC_TMR0EN		= 0x0001
.equ TC_TMR0IEN		= 0x0002
.equ TC_TMR1EN		= 0x0004
.equ TC_TMR1IEN		= 0x0008
// TMRSTATUS register
.equ TS_TMR0FLAG	= 0x0001
.equ TS_TMR1FLAG	= 0x0002

// new (SP3):
.equ TMR_BASE   = 0xc00120w
.equ TMR0CTRL   = TMR_BASE+0w
.equ TMR0STATUS = TMR_BASE+1w
.equ TMR0       = TMR_BASE+2w
.equ TMR0HI     = TMR_BASE+3w
.equ TMR1CTRL   = TMR_BASE+4w
.equ TMR1STATUS = TMR_BASE+5w
.equ TMR1       = TMR_BASE+6w
.equ TMR1HI     = TMR_BASE+7w
.equ TMR2CTRL   = TMR_BASE+8w
.equ TMR2STATUS = TMR_BASE+9w
.equ TMR2       = TMR_BASE+10w
.equ TMR2HI     = TMR_BASE+11w

.equ TC_TMREN     = 0x0001
.equ TC_TMRIRQEN  = 0x0002
.equ TC_TMRHIEN   = 0x0004
.equ TC_TMRRELOAD = 0x0008


// VLD entry formats
.equ VLD_ENTRY_FORMAT_16BIT	= 0x0		// entry has 16 bit (bit 0 not set)
.equ VLD_ENTRY_FORMAT_32BIT	= 0x1		// entry has 32 bit (bit 0 set)
.equ VLC_ENTRY_FORMAT_FLAG_EN	= 0x2		// enable decoding of carry flag
.equ VLC_ENTRY_FORMAT_SKIP_EN	= 0x4		// enable instruction skip mode
.equ VLC_ENTRY_FORMAT_SIGN_EN	= 0x8		// enable instruction sign mode

// cyclic buffer mode
.equ CBBASE_ALIGN		= 512		// alignment for cyclic buffer base address


// CAVLC accelerator unit (CVU)
.equ CVU_BASE        = 0xc00100w
.equ CVU_CTRL        = CVU_BASE+0w
.equ CVU_STATUS      = CVU_BASE+1w
.equ CVU_COEFFPTR    = CVU_BASE+2w
.equ CVU_COEFFPTR_CR = CVU_BASE+3w
.equ CVU_NA          = CVU_BASE+4w
.equ CVU_NB          = CVU_BASE+5w
.equ CVU_NA_CR422    = CVU_BASE+6w
.equ CVU_CBP         = CVU_BASE+7w
.equ CVU_CBF         = CVU_BASE+8w

.equ CVU_CTRL_START      = 0x0001
.equ CVU_CTRL_RESET      = 0x0002
.equ CVU_CTRL_ENCODE     = 0x0004
.equ CVU_CTRL_MULT       = 0x0008
.equ CVU_CTRL_SKIPDC     = 0x0010
.equ CVU_CTRL_STORETC    = 0x0020
.equ CVU_CTRL_CHROMA     = 0x0040
.equ CVU_CTRL_CHROMA422  = 0x0080
.equ CVU_CTRL_TRANSPOSE  = 0x0100
.equ CVU_CTRL_FIELDSCAN  = 0x0200
.equ CVU_CTRL_A15SEL     = 0x0400

.equ CVU_STATUS_BUSY     = 0x0001
.equ CVU_STATUS_ERROR    = 0x0002
.equ CVU_STATUS_WAITDMEM = 0x0004
.equ CVU_STATUS_WAITSBUF = 0x0008

// Processor Version
.equ SP_VERSION	= 0xc00157w
.equ GPDATA7    = 0xc00147w

// DMA channel usage (non-exclusive):
// - dma_xxx():     channel 15
// - dma_xxx_chB(): channel 13



//void *memcpy(void *dst,const void *src,size_t size)
.csection _memcpy
	.scheduling off
.export _memcpy
.export _memcpyw
_memcpy:
	and_wwi9_$	zero,r0,#31	// word aligned?
	and_wwi32_z$	zero,r1,#31	// word aligned?
	and_wwi32_z$	zero,r2,#3	// multiple of words?
	mv_wi16_z	pc,#aligned_word
	 trp		r20,a0,r0
	 trp		r21,a1,r1
	 mv		r23,zero
	 subu_cc	r23[0:31],r2[2:2]
	mv_$		r22,r2
	mv_wi16_z	skippc,#return
loop_byte:
	nop
	eloop		r22,loop_byte
	 mvu_cc		(a0)[0:8],(a1)[0:8]
	 add_wwi9	a0,a0,#8
	 add_wwi9	a1,a1,#8
	 nop
return:
	mv		a1,r21
	add_wwi9	pc,lr,#4
	 mv		a0,r20
	 nop
	 nop
	 nop

_memcpyw:	// aligned word version of memcpy
	trp		r20,a0,r0
	trp		r21,a1,r1
	mv		r23,zero
	subu_cc		r23[0:31],r2[2:2]

aligned_word:
	add		pc,r23,#aligned_4word
	 mv_$		r22,r2
	 mv_wi16_z	skippc,#return
	 lsr_$		r22,r22,#4
	 nop
	mv_ww		(-a0+),(-a1+)	// branch here to copy 3 extra words
	mv_ww		(-a0+),(-a1+)	// branch here to copy 2 extra words
	mv_ww		(-a0+),(-a1+)	// branch here to copy 1 extra words
aligned_4word:				// branch here to copy 0 extra words
	mv_wi16_z	skippc,#return
loop_4word:
	eloop		r22,loop_4word
	 mv		(-a0+),(-a1+)	// incdec may be skipped when r22==0
	 mv		(-a0+),(-a1+)	// incdec may be skipped when r22==0
	 mv		(-a0+),(-a1+)	// incdec may be skipped when r22==0
	 mv		(-a0+),(-a1+)

//return
	mv		a1,r21
	add_wwi9	pc,lr,#4
	mv		a0,r20
	nop
	nop
	nop
.endsection


.dsection dma_read_write
.export dma_read_write_buffer
	.align	1w
	dma_read_write_buffer:
	.alloc(1w)	dma_read_write_buffer_store[2] = { 0, 0 }
.endsection

.dsection dma_read_write_chB
.export dma_read_write_buffer_chB
	.align	1w
	dma_read_write_buffer_chB:
	.alloc(1w)	dma_read_write_buffer_store[2] = { 0, 0 }
.endsection

//void dma_write(unsigned addr,unsigned data);
.csection _dma_write
	.scheduling off
.export	_dma_write
_dma_write:
	wait 0x08000

	mv_dd		dma_read_write_buffer,@r1		// data to be written
	mv_dd		dma_descr.dma15.ext_byteaddr,@r0	// address to be written to
	add_wwi9	pc,lr,#4
	 mv_di25	dma_descr.dma15.int_bitaddr,#dma_read_write_buffer
	 mv_di25	dma_descr.dma15.bytelength,#4
	 mv_di25	dma_descr.dma15.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_OUT|15
.endsection

.csection _dma_write_chB
	.scheduling off
.export	_dma_write_chB
_dma_write_chB:
	wait 0x02000

	mv_dd		dma_read_write_buffer_chB,@r1		// data to be written
	mv_dd		dma_descr.dma13.ext_byteaddr,@r0	// address to be written to
	add_wwi9	pc,lr,#4
	 mv_di25	dma_descr.dma13.int_bitaddr,#dma_read_write_buffer_chB
	 mv_di25	dma_descr.dma13.bytelength,#4
	 mv_di25	dma_descr.dma13.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_OUT|13
.endsection

//void dma_write64(unsigned addr,unsigned data_hi,unsigned data_lo);
.csection _dma_write64
	.scheduling off
.export	_dma_write64
_dma_write64:
	wait 0x08000

	mv_dd		dma_read_write_buffer,@r1		// data to be written (upper 32 bit)
	mv_dd		dma_read_write_buffer+1w,@r2		// data to be written (lower 32 bit)
	mv_dd		dma_descr.dma15.ext_byteaddr,@r0	// address to be written to
	add_wwi9	pc,lr,#4
	 mv_di25	dma_descr.dma15.int_bitaddr,#dma_read_write_buffer
	 mv_di25	dma_descr.dma15.bytelength,#8
	 mv_di25	dma_descr.dma15.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_OUT|15
.endsection

.csection _dma_write64_chB
	.scheduling off
.export	_dma_write64_chB
_dma_write64_chB:
	wait 0x02000

	mv_dd		dma_read_write_buffer_chB,@r1		// data to be written (upper 32 bit)
	mv_dd		dma_read_write_buffer_chB+1w,@r2	// data to be written (lower 32 bit)
	mv_dd		dma_descr.dma13.ext_byteaddr,@r0	// address to be written to
	add_wwi9	pc,lr,#4
	 mv_di25	dma_descr.dma13.int_bitaddr,#dma_read_write_buffer_chB
	 mv_di25	dma_descr.dma13.bytelength,#8
	 mv_di25	dma_descr.dma13.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_OUT|13
.endsection


//void dma_wait(unsigned addr,unsigned data);
.csection _dma_wait
	.scheduling off
.export	_dma_wait
_dma_wait:
	wait 0x08000
	add_wwi9	pc,lr,#4
	nop
	nop
	nop
	nop
.endsection

.csection _dma_wait_chB
	.scheduling off
.export	_dma_wait_chB
_dma_wait_chB:
	wait 0x02000
	add_wwi9	pc,lr,#4
	nop
	nop
	nop
	nop
.endsection

.csection _dma_wait_all_ch
	.scheduling off
.export	_dma_wait_all_ch
_dma_wait_all_ch:
	wait 0x0a000
	add_wwi9	pc,lr,#4
	nop
	nop
	nop
	nop
.endsection

//unsigned dma_read(unsigned addr);
.csection _dma_read
	.scheduling off
.export _dma_read
_dma_read:
	wait 0x08000							// check if DMA channel 15 is ready

	mv_dd		dma_descr.dma15.ext_byteaddr,@r0	// address to be read from
	mv_di25		dma_descr.dma15.int_bitaddr,#dma_read_write_buffer
	mv_di25		dma_descr.dma15.bytelength,#4
	mv_di25		dma_descr.dma15.flags,#0
	mv_di25		@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|15

	nop
	nop

	nop
	nop
	wait 0x08000 							// check if DMA channel 15 is ready

	add_wwi9	pc,lr,#4
	 mv_dd		@r0,dma_read_write_buffer		// data to be read
	 nop
	 nop
	 nop
.endsection

.csection _dma_read_chB
	.scheduling off
.export _dma_read_chB
_dma_read_chB:
_dma_read:
	wait 0x02000							// check if DMA channel 13 is ready

	mv_dd		dma_descr.dma13.ext_byteaddr,@r0	// address to be read from
	mv_di25		dma_descr.dma13.int_bitaddr,#dma_read_write_buffer_chB
	mv_di25		dma_descr.dma13.bytelength,#4
	mv_di25		dma_descr.dma13.flags,#0
	mv_di25		@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|13

	nop
	nop

	nop
	nop
	wait 0x02000

	add_wwi9	pc,lr,#4
	 mv_dd		@r0,dma_read_write_buffer_chB		// data to be read
	 nop
	 nop
	 nop
.endsection

//void dma_write_block(unsigned ext_addr,const char *int_ptr,unsigned len);
.csection _dma_write_block
	.scheduling off
.export	_dma_write_block
_dma_write_block:
	wait 0x08000 							// check if DMA channel 15 is ready

	mv_dd		dma_descr.dma15.ext_byteaddr,@r0	// address to be written to
	add_wwi9	pc,lr,#4
	 mv_dd		dma_descr.dma15.int_bitaddr,@r1
	 mv_dd_$	dma_descr.dma15.bytelength,@r2 // prevent illegal zero size transfer
	 mv_di25	dma_descr.dma15.flags,#0
	 mv_di25_nz	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_OUT|15
.endsection

.csection _dma_write_block_chB
	.scheduling off
.export	_dma_write_block_chB
_dma_write_block_chB:
	wait 0x02000 							// check if DMA channel 13 is ready

	mv_dd		dma_descr.dma13.ext_byteaddr,@r0	// address to be written to
	add_wwi9	pc,lr,#4
	 mv_dd		dma_descr.dma13.int_bitaddr,@r1
	 mv_dd_$	dma_descr.dma13.bytelength,@r2 // prevent illegal zero size transfer
	 mv_di25	dma_descr.dma13.flags,#0
	 mv_di25_nz	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_OUT|13
.endsection


//void dma_read_block(unsigned ext_addr,char *int_ptr,unsigned len);
.csection _dma_read_block
	.scheduling off
.export _dma_read_block
_dma_read_block:

	wait 0x08000 							// check if DMA channel 15 is ready

	mv_dd		dma_descr.dma15.ext_byteaddr,@r0	// address to be read from
	mv_dd		dma_descr.dma15.int_bitaddr,@r1
	mv_dd_$		dma_descr.dma15.bytelength,@r2 // prevent illegal zero size transfer
	mv_di25		dma_descr.dma15.flags,#0
	mv_di25_nz		@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|15

	nop
	nop

	nop
	nop
	wait 0x08000							// check if DMA channel 15 is ready

	add_wwi9	pc,lr,#4
	 nop
	 nop
	 nop
	 nop
.endsection

.csection _dma_read_block_chB
	.scheduling off
.export _dma_read_block_chB
_dma_read_block_chB:

	wait 0x02000 							// check if DMA channel 13 is ready

	mv_dd		dma_descr.dma13.ext_byteaddr,@r0	// address to be read from
	mv_dd		dma_descr.dma13.int_bitaddr,@r1
	mv_dd_$		dma_descr.dma13.bytelength,@r2 // prevent illegal zero size transfer
	mv_di25		dma_descr.dma13.flags,#0
	mv_di25_nz		@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|13

	nop
	nop

	nop
	nop
	wait 0x02000 							// check if DMA channel 13 is ready

	add_wwi9	pc,lr,#4
	nop
	nop
	nop
	nop
.endsection


//void dma_read_block_nowait(unsigned ext_addr,char *int_ptr,unsigned len);
.csection _dma_read_block_nowait
	.scheduling off
.export _dma_read_block_nowait
_dma_read_block_nowait:

	wait 0x08000

	mv_dd		dma_descr.dma15.ext_byteaddr,@r0	// address to be read from
	add_wwi9	pc,lr,#4
	 mv_dd		dma_descr.dma15.int_bitaddr,@r1
	 mv_dd		dma_descr.dma15.bytelength,@r2
	 mv_di25	dma_descr.dma15.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|15
.endsection

.csection _dma_read_block_nowait_chB
	.scheduling off
.export _dma_read_block_nowait_chB
_dma_read_block_nowait_chB:

	wait 0x02000

	mv_dd		dma_descr.dma13.ext_byteaddr,@r0	// address to be read from
	add_wwi9	pc,lr,#4
	 mv_dd		dma_descr.dma13.int_bitaddr,@r1
	 mv_dd		dma_descr.dma13.bytelength,@r2
	 mv_di25	dma_descr.dma13.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|13
.endsection


//void dma_memset_ext(unsigned ext_addr,char c,unsigned len);
.csection _dma_memset_ext
	.scheduling off
.export	_dma_memset_ext
_dma_memset_ext:
	mv_ww		r20,r0
	mv_ww		r22,r2

_wait:
	nop
	nop
	wait 0x08000

	mv_dd		dma_descr.dma15.int_bitaddr,@r1

	minu_wwi32_$	r23,r22,#32768
	mv_wi16_z	skippc,#finished

	mv_dd		dma_descr.dma15.ext_byteaddr,@r20	// address to be written to
	mv_dd		dma_descr.dma15.bytelength,@r23
	mv_wi16		pc,#_wait
	 mv_di25	dma_descr.dma15.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_OUT|DC_CONST|15
	 sub_www	r22,r22,r23				// decrease remaining bytes
	 add_www	r20,r20,r23				// increase start address

finished:
	add_wwi9	pc,lr,#4
	 nop
	 nop
	 nop
	 nop
.endsection

.csection _dma_memset_ext_chB
	.scheduling off
.export	_dma_memset_ext_chB
_dma_memset_ext_chB:
	mv_ww		r20,r0
	mv_ww		r22,r2

_wait:
	nop
	nop
	wait 0x02000

	mv_dd		dma_descr.dma13.int_bitaddr,@r1

	minu_wwi32_$	r23,r22,#32768
	mv_wi16_z	skippc,#finished

	mv_dd		dma_descr.dma13.ext_byteaddr,@r20	// address to be written to
	mv_dd		dma_descr.dma13.bytelength,@r23
	mv_wi16		pc,#_wait
	 mv_di25	dma_descr.dma13.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_OUT|DC_CONST|13
	 sub_www	r22,r22,r23				// decrease remaining bytes
	 add_www	r20,r20,r23				// increase start address

finished:
	add_wwi9	pc,lr,#4
	 nop
	 nop
	 nop
	 nop
.endsection


//void dma_memset_int(char* int_ptr,char c,unsigned len);
.csection _dma_memset_int
	.scheduling off
.export	_dma_memset_int
_dma_memset_int:

	wait 0x08000

	mv_dd		dma_descr.dma15.int_bitaddr,@r0
	add_wwi9	pc,lr,#4
	 mv_dd		dma_descr.dma15.ext_byteaddr,@r1
	 mv_dd		dma_descr.dma15.bytelength,@r2
	 mv_di25	dma_descr.dma15.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|DC_CONST|15
.endsection

.csection _dma_memset_int_chB
	.scheduling off
.export	_dma_memset_int_chB
_dma_memset_int_chB:

	wait 0x02000

	mv_dd		dma_descr.dma13.int_bitaddr,@r0
	add_wwi9	pc,lr,#4
	 mv_dd		dma_descr.dma13.ext_byteaddr,@r1
	 mv_dd		dma_descr.dma13.bytelength,@r2
	 mv_di25	dma_descr.dma13.flags,#0
	 mv_di25	@DMA_CTRL,#DC_CLEAR_AUTO|DC_DIR_IN|DC_CONST|13
.endsection


// implicit C compiler function: r0=r0/r1 (unsigned)
.csection ___udivsi3
	.scheduling off
.export ___udivsi3
___udivsi3:
	cmb_www		r22,zero,r0	// dividend
	cmb_www		r23,zero,r1	// divisor
	sub_www		r23,r23,r22	// number of bits to shift divisor to match dividend MSB
	mv_ww		r22,mone	// prepare result register
	maxs_www	r23,r23,zero	// prevent underflow if dividend < divisor
	lsl_www		r21,r1,r23
	add_ww_$	r23,one		// number of loops = number of shifts + 1; also clear carry
loop:	// note: carry cleared by last instruction
	sub_ww_y	r0,r21		// subtract if overflow occured on last iteration
	addc_ww		r22,r22		// shift result and add new temporary bit - result must be finally inverted
	eloop_wi14	r23,loop
	 sub_ny$	zero,r0,r21	// try to subtract divisor from dividend
	 sub_ww_ny	r0,r21		// subtract if no overflow occurs
	 xor_ww_y	r22,one		// invert new bit
	 add_ww_$	r0,r0		// shift left dividend and save carry
	add_wwi9	pc,lr,#4
	 xor_www	r0,r22,mone	// invert result
	 nop
	 nop
	 nop
.endsection

// implicit C compiler function: r0=r0/r1 (int)
.csection ___divsi3
	.scheduling off
.export ___divsi3
___divsi3:
	mv_ww		(-sp+),lr 	// save return address
	trp_wwi32	lr,pc,___udivsi3
	 xor_www	r24,r0,r1	// save sign for result
	 mv_ww		r25,r1		// save r1 for restoring
	 abs_ww		r0,r0
	 abs_ww		r1,r1
	add_wwi9	pc,(-sp+),#4	// return address
	 lsr_wwi5_$	r24,r24,#31
	 mul_nz	r0,r0,mone		// optionally negate result
	 mv_ww		r1,r25		// restore r1
	 nop
.endsection

// implicit C compiler function: r0=r0%r1 (unsigned)
.csection ___umodsi3
	.scheduling off
.export ___umodsi3
___umodsi3:
	cmb_www		r22,zero,r0	// dividend
	cmb_www		r23,zero,r1	// divisor
	sub_www		r23,r23,r22	// number of bits to shift divisor to match dividend MSB
	maxs_www	r23,r23,zero	// prevent underflow if dividend < divisor
	lsl_www		r24,r1,r23
	mv_ww		r22,r23		// save shift to shift back result
	add_ww_$	r23,one		// number of loops = number of shifts + 1; also clear carry
loop:	// note: carry cleared by last instruction
	sub_ww_y	r0,r24		// subtract if overflow occured on last iteration
	eloop_wi14	r23,loop
	 sub_ny$	zero,r0,r24	// try to subtract divisor from dividend
	 sub_ww_ny	r0,r24		// subtract if no overflow occurs
	 sub_ww_$	zero,r23	// check for last iteration
	 add_ww_ne$	r0,r0		// shift left dividend and save carry - but not in last iteration
	add_wwi9	pc,lr,#4
	 lsr_www	r0,r0,r22
	 nop
	 nop
	 nop
.endsection

// implicit C compiler function: r0=r0%r1 (int)
.csection ___modsi3
	.scheduling off
.export ___modsi3
___modsi3:
	mv_ww		(-sp+),lr 	// save return address
	trp_wwi32	lr,pc,___umodsi3
	 mv_ww		r25,r0		// save dividend sign for result
	 mv_ww		r26,r1		// save r1 for restoring
	 abs_ww		r0,r0
	 abs_ww		r1,r1
	add_wwi9	pc,(-sp+),#4	// return address
	 lsr_wwi5_$	r25,r25,#31
	 mulss_nz	r0,r0,mone	// optionally negate result
	 mv_ww		r1,r26		// restore r1
	 nop
.endsection
