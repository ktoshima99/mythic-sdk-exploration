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
 * FILENAME: target_lowlevel_consts.h
 *
 * DESCRIPTION: Memory Map definitions for videantis LowLevel API
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef _TARGET_LOWLEVEL_CONSTS_H_
#define _TARGET_LOWLEVEL_CONSTS_H_

/**
 * @brief videantis LowLevel Library target_lowlevel_consts.h include file
 *
 * @details
 * This file contains Memory Map definitions for videantis LowLevel API
 * for core firmware sizes, mailbox layout, overlay memory, ocmem memory
 *
 * Memory Map Defaults (Overview)
 *
 * VLL_SDRAM_START : MP/SP firmware (10 * 128kB)
 * VLL_SDRAM_START + 1280 kB: Overlay Memory (32MB)
 * VLL_OCSRAM_START: Mailboxes (32 kB)
 *
 * @file target_lowlevel_consts.h
 */


/* ----- Default Firmware Placement for Simulation ------------- */

/* Size of firmware slots: 128 kB(v-MP/v-SP) */
#define VLL_MP_FW_SIZE  ( 0x00020000u )
#define VLL_SP_FW_SIZE  ( 0x00020000u )

#define VLL_MP_FW_BASE  ( VLL_SDRAM_START )
#define VLL_SP_FW_BASE  ( VLL_SDRAM_START + VLL_NUM_MP * VLL_MP_FW_SIZE )

/* calculated FW base addresses */
#define VLL_MP_0_BASE   (VLL_MP_FW_BASE + (0u * VLL_MP_FW_SIZE) )
#define VLL_MP_1_BASE   (VLL_MP_FW_BASE + (1u * VLL_MP_FW_SIZE) )
#define VLL_MP_2_BASE   (VLL_MP_FW_BASE + (2u * VLL_MP_FW_SIZE) )
#define VLL_MP_3_BASE   (VLL_MP_FW_BASE + (3u * VLL_MP_FW_SIZE) )
#define VLL_MP_4_BASE   (VLL_MP_FW_BASE + (4u * VLL_MP_FW_SIZE) )
#define VLL_MP_5_BASE   (VLL_MP_FW_BASE + (5u * VLL_MP_FW_SIZE) )
#define VLL_MP_6_BASE   (VLL_MP_FW_BASE + (6u * VLL_MP_FW_SIZE) )
#define VLL_MP_7_BASE   (VLL_MP_FW_BASE + (7u * VLL_MP_FW_SIZE) )
#define VLL_SP_0_BASE   (VLL_SP_FW_BASE + (0u * VLL_SP_FW_SIZE) )
#define VLL_SP_1_BASE   (VLL_SP_FW_BASE + (1u * VLL_SP_FW_SIZE) )



/*
 * ------------------------- Mailbox System ------------------
 */

#define VLL_MSG_MAX_SIZE         (64u)
#define VLL_MSG_PER_BOX          ( 4u)
#define VLL_MBOX_SIZE            (VLL_MSG_MAX_SIZE * VLL_MSG_PER_BOX)
#define VLL_MBOXPAIR_OFFSET      (VLL_MBOX_SIZE * 2u)
#define VLL_MBOX_IDX_MOD         (16u)

/* mbox system start address */
#define VLL_MBOX_BASE_DEFAULT    VLL_OCSRAM_START

/* calculated message box addresses */

#define VLL_MBOX_MP_0_SP_0_BASE  ( 0u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_1_SP_0_BASE  ( 1u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_2_SP_0_BASE  ( 2u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_3_SP_0_BASE  ( 3u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_4_SP_0_BASE  ( 4u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_5_SP_0_BASE  ( 5u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_6_SP_0_BASE  ( 6u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_7_SP_0_BASE  ( 7u * VLL_MBOXPAIR_OFFSET)

#define VLL_MBOX_MP_0_SP_1_BASE  ( 8u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_1_SP_1_BASE  ( 9u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_2_SP_1_BASE  (10u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_3_SP_1_BASE  (11u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_4_SP_1_BASE  (12u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_5_SP_1_BASE  (13u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_6_SP_1_BASE  (14u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_7_SP_1_BASE  (15u * VLL_MBOXPAIR_OFFSET)

#define VLL_MBOX_MP_0_HOST_BASE  (16u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_1_HOST_BASE  (17u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_2_HOST_BASE  (18u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_3_HOST_BASE  (19u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_4_HOST_BASE  (20u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_5_HOST_BASE  (21u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_6_HOST_BASE  (22u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_MP_7_HOST_BASE  (23u * VLL_MBOXPAIR_OFFSET)

#define VLL_MBOX_SP_0_SP_1_BASE  (24u * VLL_MBOXPAIR_OFFSET)
#define VLL_MBOX_SP_0_HOST_BASE  (25u * VLL_MBOXPAIR_OFFSET)

#define VLL_MBOX_SP_1_HOST_BASE  (26u * VLL_MBOXPAIR_OFFSET)

#define VLL_MBOX_COUNT 27u


/* default value: 13kB */
#define VLL_MBOXES_TOTAL_SIZE         (VLL_MBOX_COUNT * VLL_MBOXPAIR_OFFSET)

/* ----- Overlay memory -------- */
#define VLL_OVL_BASE_HOST    (VLL_SDRAM_START + VLL_NUM_MP * VLL_MP_FW_SIZE + VLL_NUM_SP * VLL_SP_FW_SIZE)
#define VLL_OVL_SIZE_HOST    (VLL_SDRAM_SIZE  - (VLL_NUM_MP * VLL_MP_FW_SIZE + VLL_NUM_SP * VLL_SP_FW_SIZE))

/* --- On chip SRAM for Host, (2MB - 13kB) OCM managed by host  */
#define VLL_OCSRAM_BASE_HOST (VLL_OCSRAM_START + VLL_MBOXES_TOTAL_SIZE)
#define VLL_OCSRAM_SIZE_HOST (VLL_OCSRAM_SIZE  - VLL_MBOXES_TOTAL_SIZE)

#endif // defined(_TARGET_LOWLEVEL_CONST_H_)
