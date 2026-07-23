/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2025 videantis GmbH
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
 * FILENAME: vid_vmp_boot_loader.asm
 *
 * DESCRIPTION: videantis v-MP 4.x bootloader with no overlay initialization
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// Stack pointer and link register definition
.equ lr = SR31
.equ sp = SFIR3

// DMA configuration registers
.equ BIU_BASE                   = 0x3B0
.equ BIU_DMA_DESCR_BASE         = {BIU_BASE + 0x0}
.equ BIU_DMA_CTRL               = {BIU_BASE + 0x1}
.equ BIU_DMA_STATUS             = {BIU_BASE + 0x2}
.equ BIU_BOOT_ADDRESS_L         = {BIU_BASE + 0xE}
.equ BIU_BOOT_ADDRESS_H         = {BIU_BASE + 0xF}

.equ BOOT_DMA_CHANNELS          = 4
.equ BOOT_DMA_CHANNELS_WORDS    = {4 * BOOT_DMA_CHANNELS}

// check if memory for llvm stack dsection is defined (default dmem2)
.if !defined(VMP_LLVM_STACK_MEM)
.equ VMP_LLVM_STACK_MEM = 2
.endif

// check if memory position for llvm stack dsection is defined (default start of dmem2: 0x400)
.if !defined(VMP_LLVM_STACK_ORG)
.equ VMP_LLVM_STACK_ORG = 0x400
.endif

// check the memory used for llvm stack and locate the boot dma channels there
.if {VMP_LLVM_STACK_MEM == 1}
.equ BOOT_DMA_CHANNELS_WORDS_DMEM  = BOOT_DMA_CHANNELS_WORDS
.equ LLVM_STACK_MEM=dmem
.else
.equ BOOT_DMA_CHANNELS_WORDS_DMEM  = 0
.endif
.if {VMP_LLVM_STACK_MEM == 2}
.equ BOOT_DMA_CHANNELS_WORDS_DMEM2 = BOOT_DMA_CHANNELS_WORDS
.equ LLVM_STACK_MEM=dmem2
.else
.equ BOOT_DMA_CHANNELS_WORDS_DMEM2 = 0
.endif
.if {VMP_LLVM_STACK_MEM == 3}
.equ BOOT_DMA_CHANNELS_WORDS_DMEM3 = BOOT_DMA_CHANNELS_WORDS
.equ LLVM_STACK_MEM=dmem3
.else
.equ BOOT_DMA_CHANNELS_WORDS_DMEM3 = 0
.endif

/* ========================================================================== *
 *         Boot code segment                                                  *
 * ========================================================================== */
.csection init
    .equ VID_IMEM_START_ADDR_AFTER_INITIAL_CODE = 0x00000030
    .equ VID_imem_WORDSIZE                      = 0x1000
    // Code length of application code without overlays (number of 64-bit words, max 0x800)
    .equ VID_IMEM_SIZE_RESIDENT_CODE        = {VIDASM_imem_LAST_NON_OVERLAY_ADDRESS + 1}

    .equ VID_IMEM_LENGTH_AFTER_INITIAL_CODE = {VID_IMEM_SIZE_RESIDENT_CODE - VID_IMEM_START_ADDR_AFTER_INITIAL_CODE}

    .equ VID_IMEM_BOOTBIN_EXTERNAL_SIZE     = {{8 * VID_IMEM_SIZE_RESIDENT_CODE} + VIDASM_OLM_DATA_EXTERNAL_OFFSET}

    .equ VID_DMEM_START                     = {0x0 + BOOT_DMA_CHANNELS_WORDS_DMEM}
    .equ VID_DMEM_LENGTH                    = {{VIDASM_dmem_LAST_NON_OVERLAY_ADDRESS + 1} - VID_DMEM_START}

    .equ VID_DMEM_BOOTBIN_EXTERNAL_SIZE     = {8 * {VID_DMEM_LENGTH + BOOT_DMA_CHANNELS_WORDS_DMEM}}

    .equ VID_DMEM2_START                    = {0x400 + BOOT_DMA_CHANNELS_WORDS_DMEM2}
    .equ VID_DMEM2_LENGTH                   = {{VIDASM_dmem2_LAST_NON_OVERLAY_ADDRESS + 1} - VID_DMEM2_START}

    .equ VID_DMEM2_BOOTBIN_EXTERNAL_SIZE     = {8 * {VID_DMEM2_LENGTH + BOOT_DMA_CHANNELS_WORDS_DMEM2}}

    .equ VID_DMEM3_START                    = {0x1000 + BOOT_DMA_CHANNELS_WORDS_DMEM3}
    .equ VID_DMEM3_LENGTH                   = {{VIDASM_dmem3_LAST_NON_OVERLAY_ADDRESS + 1} - VID_DMEM3_START}

    // Boot loader code will be aligned to address 0x0000 in I-Mem
    .org 0x0000
//	.type	vmp_bootloader, @function

vmp_bootloader:
    // *** start of GTL initialization
    // GTL simulation pipeline clean-up
.scheduling off
    V_NOP ; MV VR0, ZERO
    MVI BIU_DMA_DESCR_BASE, #llvm_runtime_stack.stack ; MVFIRI SFIR0, #llvm_runtime_stack.stack   // FIXME: won't work with dmem3 as boot dma segment (MVI instruction cannot write dmem3 immediate addresses)
    V_ADDCS_U8 VR1, VR0, VR0 ; V_ADD_U8 VR2, VR0, VR0
.scheduling on
    // *** end of GTL initialization


    // Setup up to 4 DMA transfers for code, and data segments

    // Get external boot address provided to core before booting
    MV  SR0, BIU_BOOT_ADDRESS_L

    // Load resident code into instruction memory using DMA channel 0
    // Instruction memory starts at 0xF000
    // Do not load initial 48 words already loaded by HW after booting
    // Note: this code supports only address arithmetic on the lower 32-bit of the boot address
    MV  STORE_HIGH, BIU_BOOT_ADDRESS_H
    MVI SR1, #{VID_IMEM_START_ADDR_AFTER_INITIAL_CODE * 8}
    ADD (SFIR0)+, SR0, SR1
    MVIL STORE_HIGH, #{VID_IMEM_START_ADDR_AFTER_INITIAL_CODE + 0xF000}
    MVIL (SFIR0)+, #VID_IMEM_LENGTH_AFTER_INITIAL_CODE
    // Note: the previously stored values in STORE_HIGH will be reused (write to SEGMENT_STRIDE_INT),
    //       but ignored by DMA unit (SEGMENT_LENGTH_INT is 0 => SEGMENT_STRIDE_INT will be ignored)
    MV  (SFIR0)++, ONE
    .dependency dmem3, 0
    .dependency dmem2, 0
    .dependency dmem, 0
    MVI BIU_DMA_CTRL, #{0 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}

    // Note: not preloading data memories will break any initialized dsections

    // Load data memory segment int data memory using DMA channel 1, if enabled
.if !defined(VID_VMP_BOOTLOADER_DO_NOT_PRELOAD_DMEM)
    MV  STORE_HIGH, BIU_BOOT_ADDRESS_H
    MVIL SR1, #{VID_IMEM_BOOTBIN_EXTERNAL_SIZE  + {BOOT_DMA_CHANNELS_WORDS_DMEM * 8}}
    ADD (SFIR0)+, SR0, SR1

    MVI  STORE_HIGH, #VID_DMEM_START
    MVIL (SFIR0)+, #VID_DMEM_LENGTH
    // Note: the previously stored values in STORE_HIGH will be reused (write to SEGMENT_STRIDE_INT),
    //       but ignored by DMA unit (SEGMENT_LENGTH_INT is 0 => SEGMENT_STRIDE_INT will be ignored)
    MV  (SFIR0)++, ONE
    .dependency dmem3, 0
    .dependency dmem2, 0
    .dependency dmem, 0
    MVI BIU_DMA_CTRL, #{1 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
.else
    ADDFIRI SFIR0, SFIR0, #4
.endif

    // Load data memory 2 segment int data memory 2 using DMA channel 2, if enabled
.if !defined(VID_VMP_BOOTLOADER_DO_NOT_PRELOAD_DMEM2)
    MV  STORE_HIGH, BIU_BOOT_ADDRESS_H
    MVIL SR1, #{VID_IMEM_BOOTBIN_EXTERNAL_SIZE + VID_DMEM_BOOTBIN_EXTERNAL_SIZE + {BOOT_DMA_CHANNELS_WORDS_DMEM2 * 8}}
    ADD (SFIR0)+, SR0, SR1

    MVI  STORE_HIGH, #VID_DMEM2_START
    MVIL (SFIR0)+, #VID_DMEM2_LENGTH
    // Note: the previously stored values in STORE_HIGH will be reused (write to SEGMENT_STRIDE_INT),
    //       but ignored by DMA unit (SEGMENT_LENGTH_INT is 0 => SEGMENT_STRIDE_INT will be ignored)
    MV   (SFIR0)++, ONE
    .dependency dmem3, 0
    .dependency dmem2, 0
    .dependency dmem, 0
    MVI BIU_DMA_CTRL, #{2 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
.else
    ADDFIRI SFIR0, SFIR0, #4
.endif

    // Load data memory 3 segment int data memory 3 using DMA channel 3, if enabled
.if !defined(VID_VMP_BOOTLOADER_DO_NOT_PRELOAD_DMEM3)
    MV  STORE_HIGH, BIU_BOOT_ADDRESS_H
    MVIL SR1, #{VID_IMEM_BOOTBIN_EXTERNAL_SIZE + VID_DMEM_BOOTBIN_EXTERNAL_SIZE + VID_DMEM2_BOOTBIN_EXTERNAL_SIZE + {BOOT_DMA_CHANNELS_WORDS_DMEM3 * 8}}
    ADD (SFIR0)+, SR0, SR1

    MVIL STORE_HIGH, #VID_DMEM3_START
    MVIL (SFIR0)+, #VID_DMEM3_LENGTH
    // Note: the previously stored values in STORE_HIGH will be reused (write to SEGMENT_STRIDE_INT),
    //       but ignored by DMA unit (SEGMENT_LENGTH_INT is 0 => SEGMENT_STRIDE_INT will be ignored)
    MV  (SFIR0), ONE
    .dependency dmem3, 0
    .dependency dmem2, 0
    .dependency dmem, 0
    MVI BIU_DMA_CTRL, #{3 | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD}
.endif

    // Initialize all vector registers
.if !defined(VID_VMP_BOOTLOADER_DO_NOT_INITIALIZE_REGISTERS)
    V_MVFIRI VFIR0, #@VR0
    MVI SR1, #{64 / 4}

initloop_vr:
    V_MVI_32 (VFIR0 @vreg)+, #0
    V_MVI_32 (VFIR0 @vreg)+, #0
    V_MVI_32 (VFIR0 @vreg)+, #0
    V_MVI_32 (VFIR0 @vreg)+, #0

    MVFIRI SFIR0, #@SR1
    MVI SR0, #{30 / 3}

    ELOOPR SR1, initloop_vr

    // Initialize all scalar registers
.scheduling off
initloop_sr:
    ELOOPR  SR0, initloop_sr;   MVI     (SFIR0 @sreg)+, #0
    MVI     (SFIR0 @sreg)+, #0; MVI     (SFIR0 @sreg)+, #0
.scheduling on
.endif

    // Wait for all transfers to be finished
    WAIT    #0xffff

    // Set DMA descriptor base address
    MVIL BIU_DMA_DESCR_BASE, #dma_descr.start

    MVI SR3, #_main
    // FIXME: the assembler has no SOC for a chip with v-MP 4.0 available
    // Because of this the check will fail and no program code will be executed
.if {0}
    MVI SCONDSEL, #COND_NZ
    MVFIRI SFIR0, #MP_VERSION
    SRI SR0, (SFIR0), #VIDASM_PID2
    MVIL SR1, #VIDASM_PID1
    SUBCS SR0, SR0, SR1

    MVICR GPDATA7, #0x0f
    MVICR SR3, #_exit
.endif

    // Initialize stackpointer
    MVFIRI sp, #{llvm_runtime_stack.stack + VMP_LLVM_STACK_SIZE}

    // Continue at main after bootloader finished; never return from there
    JLA ZERO, SR3
    //	.endfunction
.endsection

/* ========================================================================== *
 *         LLVM STACK                                                         *
 * ========================================================================== */
// set VMP_LLVM_STACK_SIZE based on VID_VMP_LLVM_STACK_SIZE (deprecated) or VMP_LLVM_STACK_SIZE, if nothing is set defaults to 40 words
// VID_VMP_LLVM_STACK_SIZE has the highest precedence, if VID_VMP_LLVM_STACK_SIZE is not defined VMP_LLVM_STACK_SIZE is evaluated
.if !defined(VID_VMP_LLVM_STACK_SIZE)
.if !defined(VMP_LLVM_STACK_SIZE)
.equ VMP_LLVM_STACK_SIZE = 40
.endif
.else
.equ VMP_LLVM_STACK_SIZE = VID_VMP_LLVM_STACK_SIZE
.endif

.dsection llvm_runtime_stack, LLVM_STACK_MEM
.org {VMP_LLVM_STACK_ORG}
stack:
    .alloc[VMP_LLVM_STACK_SIZE]

    // Reserve space for DMA descriptors for bootloader in case of small stack
.if {VMP_LLVM_STACK_SIZE < BOOT_DMA_CHANNELS_WORDS}
    .alloc dma_descr_tmp[BOOT_DMA_CHANNELS_WORDS - VMP_LLVM_STACK_SIZE]
.endif

.endsection

.end
