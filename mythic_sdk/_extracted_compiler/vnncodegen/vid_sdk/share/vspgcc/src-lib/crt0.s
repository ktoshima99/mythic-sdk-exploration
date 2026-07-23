//++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++
// (c) videantis GmbH
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// FILENAME:             $RCSfile$
//
//--------------------------------------------------------------------------
//
// RESPONSIBLE:         Boris Boesler
//
// DESCRIPTION:         v-SP2 assembler test file
//
// Id:                  $Id$
//
// CREATED:             30.11.2007
//
// LAST CHECKED IN BY:  $Author$
//
// NOTES:
//
// MODIFIED:
//
// $Log$
//
//+++++++++++++++++++++++++++++ FileHeaderEnd ++++++++++++++++++++++++++++++

// some values for debugging @ videantis
	.equ	PRE_INITIALIZE_ARGUMENTS = 0
// store result if neccassary
	.equ	STORE_RESULT_IN_MEMORY = 01
// address to store result
	.equ	STORE_RESULT_ADDRESS = 0
// bitstreaming register
	.equ	STREAM_REGNUM = 15
// stack-pointer address register number
	.equ	SP_REGNUM = 13
// frame-pointer address register number
	.equ	FP_REGNUM = 12
// return address data register number
	.equ	RA_REGNUM = 31
// return value from main() and exit() [R0 is pre-allocated by Hardware-System]
	.equ	RETURN_VALUE_REGNUM = 31
// do we use functional simulation?
	.equ	SP2GCC_USE_FUNCTIONAL_SIMULATION = false
	.equ	SP2GCC_STACK_GROWS_DOWNWARD = false

	// stack address
	.if { SP2GCC_STACK_GROWS_DOWNWARD }
	.equ	SP2GCC_STACK_ADDR = @DMEM_END
	.else
	// convert 32-bit addr to bit addr and align to 32 bit
	.equ	SP2GCC_STACK_ADDR = { 32 * {VIDASM_dmem_LAST_USED_ADDRESS + 1} }
	.endif

	//
	// include default constant values
	//
	.include "sp2_defs.inc"

	//
	// include (hard-core) debugging macro package
	//
	//.include "debugging-macro.inc"

// **********************************************************************
// code section
// **********************************************************************

	//
	// main entry point for generated C code in functional model
	//
	.csection CODE
        .org	0w
	.export	_exit
	
	.scheduling off
	
	//
	// generic set up of processor and registers
	// setup cyclic buffer address mode -> make Ai linear
	//
init_regs:
	.loop i = [0..15]
	MV_DD		@A[i].CBBASE,     @ZERO
	MV_DD		@A[i].CBRANGE,    @MONE
	MV_DD		@A[i].CBSUBRANGE, @MONE
	MV_WI16		A[i].INCDEC,	#1w
	.endloop

	//
	// set up stackpointer
init_stackpointer:
	//
	MV_DD		@A[SP_REGNUM].CBBASE,     @ZERO
	MV_DD		@A[SP_REGNUM].CBRANGE,    @MONE
	MV_DD		@A[SP_REGNUM].CBSUBRANGE, @MONE
	.if { SP2GCC_STACK_GROWS_DOWNWARD }
	MV_WI16		A[SP_REGNUM].INCDEC,	#1w
	.else
	MV_WI16		A[SP_REGNUM].INCDEC,	#-1w
	.endif

	MV_OI32		A[SP_REGNUM], # SP2GCC_STACK_ADDR
	
	// setup MEMCFG registers to stack mode for A13
	AND_WWI32	memcfg1, memcfg1, #~{MCFG_ALL   << {MCFG_SLICE_WIDTH*{SP_REGNUM-8}}}
	OR_WWI32	memcfg1, memcfg1, # {MCFG_STACK << {MCFG_SLICE_WIDTH*{SP_REGNUM-8}}}
	// setup MEMCFG registers to streaming mode for A15
	AND_WWI32	memcfg1, memcfg1, #~{MCFG_ALL   << {MCFG_SLICE_WIDTH*{STREAM_REGNUM-8}}}
	OR_WWI32	memcfg1, memcfg1, # {MCFG_STREAM << {MCFG_SLICE_WIDTH*{STREAM_REGNUM-8}}}
	
	// for some debugging initialize args with values
	.if { PRE_INITIALIZE_ARGUMENTS }
	MV_WI16		R0, #20
	MV_WI16		R1, #22	// 20 + 22 = 42 ;-)
	.else
	// argc := 1
        // argv := NULL
	MV_WW		R0, ONE
	MV_WW		R1, ZERO
	.endif

	//
	// call main routine
	// set arguments R0 and R1 to 0
	//
call_main:
	MV_OI32		R[RA_REGNUM], #_main
	JLA_WW		R[RA_REGNUM], R[RA_REGNUM]
	NOP
	NOP
	NOP
	NOP

_exit:
	// move return value from default register into special register
	MV_WW	R[RETURN_VALUE_REGNUM], R0
	
	.if { STORE_RESULT_IN_MEMORY }
	// address to store result
	MV_DD	STORE_RESULT_ADDRESS, @R0
	.endif

	.if SP2GCC_USE_FUNCTIONAL_SIMULATION

	.call terminate

	.else
	// check if all DMA channels are ready
	MV_DD		@R1,@DMA_STATUS
	SUBS_CC_$	MONE,R1[0:16]
	MV_WI16_NZ	SKIPPC, #_exit
	
	MV_OI32		R0, #0x1FFF0004			// exit address (chip-it compatible)
	MV_DI25		@DMA_DESCR_BASE, #simulator_dma_descriptor.start
	MV_DD		simulator_dma_descriptor.ext_byteaddr, @R0
	MV_DI25		simulator_dma_descriptor.int_bitaddr, #0
	MV_DI25		simulator_dma_descriptor.bytelength, #4
	MV_DI25		simulator_dma_descriptor.flags, #0
	MV_DI25		@DMA_CTRL, #{{1<<5} | {0<<4} | 0}

endless:
	MV_WI16		PC, #endless			// hardware will loop forever
	 NOP
	 NOP
	 NOP
	 NOP

	.endif

	NOP

	.scheduling on
        .endsection

// **********************************************************************
// data section
// **********************************************************************

        .dsection DATA
	
// if we store the result of main,
// then we store it at address 0, so keep it free
// else we keep address 0 free for C constructs ala if(NULL != adr) ...

	.if { STORE_RESULT_IN_MEMORY }

	.org STORE_RESULT_ADDRESS
	.align	32
result_address:
	.alloc(32bit)	NULL[1] = { 0x0 }

	.endif

	.endsection


	// dma descriptor to stop simulator
	.dsection simulator_dma_descriptor
	// for 256 byte alignment
	.align 256 * 8
start:
	.alloc(1w)	int_bitaddr
	.alloc(1w)	ext_byteaddr
	.alloc(1w)	bytelength
	.alloc(1w)	flags
	
	.endsection

// end of file
