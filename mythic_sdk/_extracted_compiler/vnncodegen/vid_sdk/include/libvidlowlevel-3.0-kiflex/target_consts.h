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
 * FILENAME:    target_consts.h
 *
 * DESCRIPTION: Hardware-specific constants
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef _TARGET_CONSTS_H_
#define _TARGET_CONSTS_H_

/**
 * @brief videantis LowLevel Library target_consts.h include file
 *
 * @details
 * This file contains all Hardware-specific constants, Memory Map,
 * core debug and control register interface map,
 * ISP, watchdog, ECC control register maps
 *
 * @file target_consts.h
 */

/* Generic Hardware related constants */

/*
 * Default Clock Frequencies
 */
#define VLL_BUS_FREQUENCY  800u
#define VLL_SP_FREQUENCY   0u
#define VLL_MP_FREQUENCY   1000u

/*
 * Memory Map
 */
/// Start address of DDR memory
#define VLL_SDRAM_START       ( 0x0000010000820000ull )
/// Size of DDR memory 256 MByte
#define VLL_SDRAM_SIZE        ( 0x10000000u )
/// Start address of on chip SRAM
#define VLL_OCSRAM_START      ( 0x0000010000000000ull )
/// Size of och chip SRAM 2 MByte (KIFlex)
#define VLL_OCSRAM_SIZE       ( 0x00200000u )

/// Start address of Debug and Control Slave window for KIFlex
#define VMP_CTRL_BASE         ( 0x0000010000800000u )

/* Size of register windows is 4kB */
#define VMP_REGWIN_SIZE       (  0x01000u )
#define IRQ_REGWIN_SIZE       (  0x00800u )

/* Base offsets for groups of registers */
#define VMP_MP_REGWIN_BASE    (  0u * VMP_REGWIN_SIZE )
#define VMP_SP_REGWIN_BASE    ( 20u * VMP_REGWIN_SIZE )
#define AURORA_REGWIN_BASE    ( 27u * VMP_REGWIN_SIZE )
#define VMP_ISP_REGWIN_BASE   ( 29u * VMP_REGWIN_SIZE )
#define VMP_IRQ_REGWIN_BASE   ( 31u * VMP_REGWIN_SIZE )
#define EDMA_REGWIN_BASE      ( VMP_IRQ_REGWIN_BASE + IRQ_REGWIN_SIZE )

/* Derived register window bases for processor cores */
#define VMP_MP_0_REGWIN_BASE  ( VMP_MP_REGWIN_BASE + (0u * VMP_REGWIN_SIZE) )
#define VMP_MP_1_REGWIN_BASE  ( VMP_MP_REGWIN_BASE + (1u * VMP_REGWIN_SIZE) )
#define VMP_MP_2_REGWIN_BASE  ( VMP_MP_REGWIN_BASE + (2u * VMP_REGWIN_SIZE) )
#define VMP_MP_3_REGWIN_BASE  ( VMP_MP_REGWIN_BASE + (3u * VMP_REGWIN_SIZE) )
#define VMP_MP_4_REGWIN_BASE  ( VMP_MP_REGWIN_BASE + (4u * VMP_REGWIN_SIZE) )
#define VMP_MP_5_REGWIN_BASE  ( VMP_MP_REGWIN_BASE + (5u * VMP_REGWIN_SIZE) )
#define VMP_MP_6_REGWIN_BASE  ( VMP_MP_REGWIN_BASE + (6u * VMP_REGWIN_SIZE) )
#define VMP_MP_7_REGWIN_BASE  ( VMP_MP_REGWIN_BASE + (7u * VMP_REGWIN_SIZE) )

#define VMP_SP_0_REGWIN_BASE  ( VMP_SP_REGWIN_BASE + (0u * VMP_REGWIN_SIZE) )
#define VMP_SP_1_REGWIN_BASE  ( VMP_SP_REGWIN_BASE + (1u * VMP_REGWIN_SIZE) )

/* Offsets for v-MP processor register windows */
/* v-MP: windows 0 ... 7                       */
#define GPDATA0_OFFSET        ( 0x00u )
#define GPDATA1_OFFSET        ( 0x08u )
#define GPDATA2_OFFSET        ( 0x10u )
#define GPDATA3_OFFSET        ( 0x18u )
#define GPDATA4_OFFSET        ( 0x20u )
#define GPDATA5_OFFSET        ( 0x28u )
#define GPDATA6_OFFSET        ( 0x30u )
#define GPDATA7_OFFSET        ( 0x38u )

#define IRQOUT_MP_OFFSET      ( 0x80u )
#define IRQIN_MP_OFFSET       ( 0x88u )

#define IRQSTATUS_OFFSET      ( 0xa8u )
#define INSTR_LO_OFFSET       ( 0xc0u )
#define INSTR_LEFT_OFFSET     ( 0xc0u )
#define INSTR_HI_OFFSET       ( 0xc8u )
#define INSTR_RIGHT_OFFSET    ( 0xc8u )
#define PC_OFFSET             ( 0xd0u )
#define MP_STATE_OFFSET       ( 0xd8u )
#define BP_ADDR_OFFSET        ( 0xe0u )
#define BP_CTRL_OFFSET        ( 0xe8u )

/* Offsets for v-SP processor register windows */
/* v-SP: windows 20, 21                        */
//FIXME: v-SP offsets are +4 of the above v-MP offsets (except IRQ ?!?)
#define GPDATA0_SP_OFFSET     ( 0x04u )
#define GPDATA1_SP_OFFSET     ( 0x0cu )
#define GPDATA2_SP_OFFSET     ( 0x14u )
#define GPDATA3_SP_OFFSET     ( 0x1cu )
#define GPDATA4_SP_OFFSET     ( 0x24u )
#define GPDATA5_SP_OFFSET     ( 0x2cu )
#define GPDATA6_SP_OFFSET     ( 0x34u )
#define GPDATA7_SP_OFFSET     ( 0x3cu )
#define GPDATA8_SP_OFFSET     ( 0x44u )
#define GPDATA9_SP_OFFSET     ( 0x4cu )

#define IRQOUT_SP_OFFSET      ( 0x7cu )

#define BOOT_ADDR_OFFSET      ( 0x200u )
#define CORE_CTRL_OFFSET      ( 0x208u )
#define TEST_AND_SET_0_OFFSET ( 0x220u )
#define TEST_AND_SET_1_OFFSET ( 0x240u )

#define ECC_IMEM_CNT_OFFSET       ( 0x300u )
#define SRAM_CTRL_SP_GEN_OFFSET   ( 0x308u )
#define SRAM_CTRL_MP_GEN_OFFSET   ( 0x308u )
#define SRAM_CTRL_SP_IMEM_OFFSET  ( 0x310u )
#define SRAM_CTRL_MP_IMEM_OFFSET  ( 0x310u )
#define SRAM_CTRL_SP_DMEM_OFFSET  ( 0x318u )
#define SRAM_CTRL_MP_DMEM_OFFSET  ( 0x318u )
#define SRAM_CTRL_SP_PO_OFFSET    ( 0x320u )
#define SRAM_CTRL_MP_PO_OFFSET    ( 0x320u )
#define ECC_IMEM_TEST_OFFSET      ( 0x328u )

/// Watchdog timer (special access)
#define WD_TIMER_OFFSET           ( 0x0400u )
#define WD_EN_OFFSET              ( 0x0410u )
#define WD_TICK_DIV_OFFSET        ( 0x0420u )


/* Offsets for ISP Handshake Unit register window 29 */
#define ISP0_CTRL_OFFSET           ( 0x0000u )
#define ISP0_BUFFADDR0_OFFSET      ( 0x0008u )
#define ISP0_BUFFADDR1_OFFSET      ( 0x0010u )
#define ISP0_BUFFADDR2_OFFSET      ( 0x0018u )
#define ISP0_STATUS_OFFSET         ( 0x0020u )
#define ISP0_BA_OFFSET             ( 0x0028u )
#define ISP0_BUFFADDR0_S_OFFSET    ( 0x0030u )
#define ISP0_BUFFADDR1_S_OFFSET    ( 0x0038u )
#define ISP0_BUFFADDR2_S_OFFSET    ( 0x0040u )
#define ISP0_COUNT_OFFSET          ( 0x0048u )
#define ISP1_CTRL_OFFSET           ( 0x0050u )
#define ISP1_BUFFADDR0_OFFSET      ( 0x0058u )
#define ISP1_BUFFADDR1_OFFSET      ( 0x0060u )
#define ISP1_BUFFADDR2_OFFSET      ( 0x0068u )
#define ISP1_STATUS_OFFSET         ( 0x0070u )
#define ISP1_BA_OFFSET             ( 0x0078u )
#define ISP1_BUFFADDR0_S_OFFSET    ( 0x0080u )
#define ISP1_BUFFADDR1_S_OFFSET    ( 0x0088u )
#define ISP1_BUFFADDR2_S_OFFSET    ( 0x0090u )
#define ISP1_COUNT_OFFSET          ( 0x0098u )

/* Offsets for Watchdog Unit register window 30 */
#define VID_WD_MP_BASE            ( 0x000u )
#define VID_WD_SIZE               (  0x20u )
#define VID_WD_SP_BASE            ( 0x280u )


/* Offsets for IRQ Concentrator Unit register window 31 */
#define GP_IRQ_STATUS_OFFSET        ( 0x0000u )
#define GP_IRQ_MASK_OFFSET          ( 0x0008u )
#define GP_IRQ_SET_EN_OFFSET        ( 0x0010u )
#define GP_IRQ_CLEAR_EN_OFFSET      ( 0x0018u )

#define ECC_NC_IRQ_STATUS_OFFSET    ( 0x0020u )
#define ECC_NC_IRQ_CLEAR_OFFSET     ( 0x0028u )
#define ECC_NC_IRQ_MASK_OFFSET      ( 0x0030u )
#define ECC_NC_IRQ_SET_EN_OFFSET    ( 0x0038u )
#define ECC_NC_IRQ_CLEAR_EN_OFFSET  ( 0x0040u )
#define ECC_NC_IRQ_SET_OFFSET       ( 0x0048u )

#define ECC_C_IRQ_STATUS_OFFSET     ( 0x0050u )
#define ECC_C_IRQ_CLEAR_OFFSET      ( 0x0058u )
#define ECC_C_IRQ_MASK_OFFSET       ( 0x0060u )
#define ECC_C_IRQ_SET_EN_OFFSET     ( 0x0068u )
#define ECC_C_IRQ_CLEAR_EN_OFFSET   ( 0x0070u )
#define ECC_C_IRQ_SET_OFFSET        ( 0x0078u )

#define BUS_ERR_IRQ_STATUS_OFFSET     ( 0x0080u )
#define BUS_ERR_IRQ_CLEAR_OFFSET      ( 0x0088u )
#define BUS_ERR_IRQ_MASK_OFFSET       ( 0x0090u )
#define BUS_ERR_IRQ_SET_EN_OFFSET     ( 0x0098u )
#define BUS_ERR_IRQ_CLEAR_EN_OFFSET   ( 0x00A0u )
#define BUS_ERR_IRQ_SET_OFFSET        ( 0x00A8u )

#define WD_IRQ_STATUS_OFFSET        ( 0x00B0u )
#define WD_IRQ_MASK_OFFSET          ( 0x00B8u )
#define WD_IRQ_SET_EN_OFFSET        ( 0x00C0u )
#define WD_IRQ_CLEAR_EN_OFFSET      ( 0x00C8u )

#define ISP0_IRQ_STATUS_OFFSET      ( 0x0100u )
#define ISP0_IRQ_BMASK_OFFSET       ( 0x0108u )
#define ISP0_IRQ_SET_EN_OFFSET      ( 0x0110u )
#define ISP0_IRQ_CLEAR_EN_OFFSET    ( 0x0118u )

#define ISP1_IRQ_STATUS_OFFSET      ( 0x0120u )
#define ISP1_IRQ_BMASK_OFFSET       ( 0x0128u )
#define ISP1_IRQ_SET_EN_OFFSET      ( 0x0130u )
#define ISP1_IRQ_CLEAR_EN_OFFSET    ( 0x0138u )

#endif // ifndef _TARGET_CONSTS_H
