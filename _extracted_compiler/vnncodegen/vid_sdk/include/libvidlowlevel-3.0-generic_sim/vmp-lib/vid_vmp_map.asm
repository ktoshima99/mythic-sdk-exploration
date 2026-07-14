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
 * FILENAME: vid_vmp_map.asm
 *
 * DESCRIPTION: videantis v-MP register map
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP register map
 *
 * @details
 * This file provides the global register map used by the compiler vmpcc
 *
 * @file vid_vmp_map.asm
 */

/// @cond DOXYGEN_IGNORE_ASM

//
// The following register is used as stack pointer and link register
// (defined in vid_vmp_boot_loader.asm)
//
//.equ lr = r31
//.equ sp = sfir3
/* ============================================================================== */
//
// Possible Constants for both CONDSEL-Register
//
.equ COND_C = 0
.equ COND_O = 1
.equ COND_Z = 2
.equ COND_N = 3
.equ COND_P = 4
.equ COND_G = 5
.equ COND_GE = 6
.equ COND_GU = 7
.equ COND_L = 8
.equ COND_LE = 9
.equ COND_LEU = 10
.equ COND_NC = 11
.equ COND_NO = 12
.equ COND_NZ = 13
.equ COND_NV = 14  //only for VCONDSEL
.equ COND_EV = 15  //only for VCONDSEL
.equ COND_DC = 14  //only for SCONDSEL
.equ COND_DD = 15  //only for SCONDSEL
.equ COND_GEU = COND_NC
.equ COND_LU = COND_C
.equ COND_EQ = COND_Z
.equ COND_NE = COND_NZ


/* ============================================================================== */
//
// internal global address map
//
// .equ TIMER1  = 0x3d0
// .equ TCTRL1  = 0x3d1
// .equ TIMER2  = 0x3d2
// .equ TCTRL2  = 0x3d3

// big endian VPERMREG register view
.equ VPERMMASK_BE8  = 0x388
.equ VPERMMASK_BE16 = 0x389
.equ VPERMMASK_BE32 = 0x38a

// little endian VPERMREG register view
.equ VPERMMASK_LE8  = 0x38c
.equ VPERMMASK_LE16 = 0x38d
.equ VPERMMASK_LE32 = 0x38e

// big endian VPERMREG2 register view
.equ VPERMMASK2_BE8  = 0x390
.equ VPERMMASK2_BE16 = 0x391
.equ VPERMMASK2_BE32 = 0x392

// little endian VPERMREG2 register view
.equ VPERMMASK2_LE8  = 0x394
.equ VPERMMASK2_LE16 = 0x395
.equ VPERMMASK2_LE32 = 0x396

// VFIR byte address register views
.equ VFIRB0 = 0x378
.equ VFIRB1 = 0x379
.equ VFIRB2 = 0x37A
.equ VFIRB3 = 0x37B
.equ VFIRB4 = 0x37C
.equ VFIRB5 = 0x37D
.equ VFIRB6 = 0x37E
.equ VFIRB7 = 0x37F

// IRQ registers
.equ IRQIN   = 0x3d8
.equ IRQOUT  = 0x3d9

// GPDATA registers
.equ GPDATA0 = 0x3da
.equ GPDATA1 = 0x3db
.equ GPDATA2 = 0x3dc
.equ GPDATA3 = 0x3dd
.equ GPDATA4 = 0x3de
.equ GPDATA5 = 0x3df
.equ GPDATA6 = 0x3e0
.equ GPDATA7 = 0x3e1
.equ GPDFLAGS= 0x3e2

// MP_VERSION register
.equ MP_VERSION= 0x3ef

/* ============================================================================== *
 * Register definitions as used by assembly code
 *
 * If assembly functions are called from C-Code, all registers touched have
 * to be saved on stack (stackpointer: SFIR3/sp)
 *
 * The compiler can reserve registers for assembly code that don't need to
 * saved on stack with the following compiler options. These registers
 * are then not available to the compiler.
 *  -vmp-reserve-sr8-sr15
 *  -vmp-reserve-sr16-sr23
 *  -vmp-reserve-vr8-vr23
 *  -vmp-reserve-vr24-vr39
 *  -vmp-reserve-vr40-vr55
 *
 * The following registers are used for function call parameters
 *   vector parameters in the order of argument list VR0, VR1, VR2, VR3, VR4, VR5
 *   scalar parameters in the order of argument list SR0, SR1, SR2, SR3, SR4, SR5
 *   scalar function return value SR0
 *   vector function return value VR0
 *
 * The following temp registers are available for functions and compiler buitlins macro calls
 *   scalar temp registers SR27, SR28, SR29, SR30
 *   vector temp registers VR58, VR59, VR60, VR61, VR62, VR63
 *   scalar index registers SFIR0, SFIR1, SFIR2
 *   vector index registers VFIR0, VFIR1, VFIR2, VFIR3,
 *                          VFIR4, VFIR5, VFIR6, VFIR7
 *
 * The following registers can be declared as additional temp registers.
 * These registers don't need to be saved on stack with the following compiler options.
 * These registers are then also available to the compiler as temp registers.
 *  -vmp-save-callee-register
 *    =nottemporary-sr16-sr23                         -   SR16-SR23
 *    =nottemporary-sr16-sr23-vr40-vr55               -   SR16-SR23, VR40-VR55
 *    =nottemporary-sr16-sr23-vr24-vr55               -   SR16-SR23, VR24-VR55
 *    =nottemporary-sr16-sr23-vr8-vr55                -   SR16-SR23, VR8-VR55
 *    =nottemporary-sr8-sr23                          -   SR8-SR23
 *    =nottemporary-sr8-sr23-vr40-vr55                -   SR8-SR23, VR40-VR55
 *    =nottemporary-sr8-sr23-vr24-vr55                -   SR8-SR23, VR24-VR55
 *    =nottemporary-sr8-sr23-vr8-vr55                 -   SR8-SR23, VR8-VR55
 *    =nottemporary-vr40-vr55                         -   VR40-VR55
 *    =nottemporary-vr24-vr55                         -   VR24-VR55
 *    =nottemporary-vr8-vr55                          -   VR8-VR55
 *
 *
 * The following registers have to be saved when used in assembly functions
 *   scalar SR6, SR7, SR24, SR25, SR26
 *   vector VR6, VR7, VR56, VR57
 *   stack pointer          SFIR3 (cannot be saved on stack, use mbox.tmp data memory)
 *
 * ============================================================================== */

// The following registers are used for function call parameters
// vector parameters in the order of argument list
.equ vparam0 = VR0
.equ vparam1 = VR1
.equ vparam2 = VR2
.equ vparam3 = VR3
.equ vparam4 = VR4
.equ vparam5 = VR5

// The following registers are used by the compiler
// and have to be saved when used in assembly functions
.equ cvtmp0	= VR6
.equ cvtmp1	= VR7

//
//-vmp-reserve-vr8-vr23
//-vmp-save-callee-register=nottemporary-<RANGE>
//
// The following registers are used by the compiler
// and have to be saved when used in assembly functions
// With the compiler options -vmp-reserve-vr8-vr23 or
// a matching -vmp-save-callee-register=nottemporary-<RANGE>
// assembly functions can use these registers without saving
.equ vr0tmp0 	= VR8
.equ vr0tmp1 	= VR9
.equ vr0tmp2 	= VR10
.equ vr0tmp3 	= VR11
.equ vr0tmp4 	= VR12
.equ vr0tmp5 	= VR13
.equ vr0tmp6 	= VR14
.equ vr0tmp7 	= VR15
.equ vr0tmp8 	= VR16
.equ vr0tmp9 	= VR17
.equ vr0tmp10	= VR18
.equ vr0tmp11	= VR19
.equ vr0tmp12	= VR20
.equ vr0tmp13	= VR21
.equ vr0tmp14	= VR22
.equ vr0tmp15	= VR23

//
//-vmp-reserve-vr24-vr39
//-vmp-save-callee-register=nottemporary-<RANGE>
//
// The following registers are used by the compiler
// and have to be saved when used in assembly functions
// With the compiler options -vmp-reserve-vr24-vr39 or
// a matching -vmp-save-callee-register=nottemporary-<RANGE>
// assembly functions can use these registers without saving
.equ vr1tmp0 	= VR24
.equ vr1tmp1 	= VR25
.equ vr1tmp2 	= VR26
.equ vr1tmp3 	= VR27
.equ vr1tmp4 	= VR28
.equ vr1tmp5 	= VR29
.equ vr1tmp6 	= VR30
.equ vr1tmp7 	= VR31
.equ vr1tmp8 	= VR32
.equ vr1tmp9 	= VR33
.equ vr1tmp10	= VR34
.equ vr1tmp11	= VR35
.equ vr1tmp12	= VR36
.equ vr1tmp13	= VR37
.equ vr1tmp14	= VR38
.equ vr1tmp15	= VR39

//
//-vmp-reserve-vr40-vr55
//-vmp-save-callee-register=nottemporary-<RANGE>
//
// The following registers are used by the compiler
// and have to be saved when used in assembly functions
// With the compiler options -vmp-reserve-vr40-vr55 or
// a matching -vmp-save-callee-register=nottemporary-<RANGE>
// assembly functions can use these registers without saving
.equ vr2tmp0 	= VR40
.equ vr2tmp1 	= VR41
.equ vr2tmp2 	= VR42
.equ vr2tmp3 	= VR43
.equ vr2tmp4 	= VR44
.equ vr2tmp5 	= VR45
.equ vr2tmp6 	= VR46
.equ vr2tmp7 	= VR47
.equ vr2tmp8 	= VR48
.equ vr2tmp9 	= VR49
.equ vr2tmp10	= VR50
.equ vr2tmp11	= VR51
.equ vr2tmp12	= VR52
.equ vr2tmp13	= VR53
.equ vr2tmp14	= VR54
.equ vr2tmp15	= VR55

// The following registers are used by the compiler
// and have to be saved when used in assembly functions
.equ cvtmp2	= VR56
.equ cvtmp3	= VR57

// The following temp registers are available for functions and compiler buitlins macro calls
// assembly functions can use these registers without saving
.equ mvtmp0 = VR58
.equ mvtmp1 = VR59
.equ mvtmp2 = VR60
.equ mvtmp3 = VR61
.equ mvtmp4 = VR62
.equ mvtmp5 = VR63


// The following vector index registers are used by the compiler
// assembly functions can use these registers without saving
.equ vidx0 = VFIR0
.equ vidx1 = VFIR1
.equ vidx2 = VFIR2
.equ vidx3 = VFIR3
.equ vidx4 = VFIR4
.equ vidx5 = VFIR5
.equ vidx6 = VFIR6
.equ vidx7 = VFIR7

.equ vidx0B = VFIRB0
.equ vidx1B = VFIRB1
.equ vidx2B = VFIRB2
.equ vidx3B = VFIRB3
.equ vidx4B = VFIRB4
.equ vidx5B = VFIRB5
.equ vidx6B = VFIRB6
.equ vidx7B = VFIRB7

// The following scalar index registers are used by the compiler
// assembly functions can use these registers without saving
.equ sidx0		= SFIR0
.equ sidx1		= SFIR1
.equ sidx2		= SFIR2

// The following scalar index registers is used by the compiler
// as stack pointer sp. This register cannot be saved on stack.
// If an assembly functions wants to use this register, save it
// to a data memory location (e.g. use mbox.tmp)
//.equ sidx3	  = SFIR3  //defined in boot loader




// The following registers are used for function call parameters
// vector parameters in the order of argument list
.equ sparam0 = SR0
.equ sparam1 = SR1
.equ sparam2 = SR2
.equ sparam3 = SR3
.equ sparam4 = SR4
.equ sparam5 = SR5


// The following registers are used by the compiler
// and have to be saved when used in assembly functions
.equ cstmp0	= SR6
.equ cstmp1	= SR7

//
//-vmp-reserve-sr8-sr15
//-vmp-save-callee-register=nottemporary-<RANGE>
//
// The following registers are used by the compiler
// and have to be saved when used in assembly functions
// With the compiler options -vmp-reserve-sr8-sr15 or
// a matching -vmp-save-callee-register=nottemporary-<RANGE>
// assembly functions can use these registers without saving
.equ sr0tmp0	= SR8
.equ sr0tmp1	= SR9
.equ sr0tmp2	= SR10
.equ sr0tmp3	= SR11
.equ sr0tmp4	= SR12
.equ sr0tmp5	= SR13
.equ sr0tmp6	= SR14
.equ sr0tmp7	= SR15

//
//-vmp-reserve-sr16-sr23
//-vmp-save-callee-register=nottemporary-<RANGE>
//
// The following registers are used by the compiler
// and have to be saved when used in assembly functions
// With the compiler options -vmp-reserve-sr8-sr15 or
// a matching -vmp-save-callee-register=nottemporary-<RANGE>
// assembly functions can use these registers without saving
.equ sr1tmp0	= SR16
.equ sr1tmp1	= SR17
.equ sr1tmp2	= SR18
.equ sr1tmp3	= SR19
.equ sr1tmp4	= SR20
.equ sr1tmp5	= SR21
.equ sr1tmp6	= SR22
.equ sr1tmp7	= SR23

// The following registers are used by the compiler
// and have to be saved when used in assembly functions
.equ cstmp2	= SR24
.equ cstmp3	= SR25
.equ cstmp4	= SR26


// The following temp registers are available for functions and compiler buitlins macro calls
// assembly functions can use these registers without saving
.equ mstmp0 = SR27
.equ mstmp1 = SR28
.equ mstmp2 = SR29
.equ mstmp3 = SR30

// The following register are used by the compiler as link register
// and has to be saved when used in assembly functions
//.equ lr   = SR31 //link register defined in bootloader

/// @endcond
