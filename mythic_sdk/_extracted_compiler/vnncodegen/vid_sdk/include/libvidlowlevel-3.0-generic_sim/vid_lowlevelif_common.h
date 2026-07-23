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
 * FILENAME: vid_lowlevelif_common.h
 *
 * DESCRIPTION: videantis LowLevel Library common include file for v-MP and host
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis LowLevel Library common include file for v-MP and host
 *
 * @details
 * This file contains all common macros definitions
 * of videantis Lowlevel Library shared by v-MP and host
 *
 * @file vid_lowlevelif_common.h
 */

//lint -save

#ifndef __VID_LOWLEVELIF_COMMON_H__
#define __VID_LOWLEVELIF_COMMON_H__

/// Numeric Lowlevel Library version
#define VLL_VERSION       30u
/// Lowlevel Library version major
#define VLL_VERSION_MAJOR  3u
/// Lowlevel Library version minor
#define VLL_VERSION_MINOR  0u

/// Number of v-MP cores in subsystem
#define VLL_NUM_MP      8u
/// Number of v-SP cores in subsystem
#define VLL_NUM_SP      2u
/// Total number of cores in subsystem
#define VLL_NUM_CORES   10u
/// Smallest value for core IDs
#define VLL_ID_MIN      0u
/// Biggest value for core IDs
#define VLL_ID_MAX      32u

/// Core ID for v-MP #0
#define VLL_ID_MP_0     0u
/// Core ID for v-MP #1
#define VLL_ID_MP_1     1u
/// Core ID for v-MP #2
#define VLL_ID_MP_2     2u
/// Core ID for v-MP #3
#define VLL_ID_MP_3     3u
/// Core ID for v-MP #4
#define VLL_ID_MP_4     4u
/// Core ID for v-MP #5
#define VLL_ID_MP_5     5u
/// Core ID for v-MP #6
#define VLL_ID_MP_6     6u
/// Core ID for v-MP #7
#define VLL_ID_MP_7     7u

/// Smallest value for v-MP core IDs
#define VLL_ID_MP_MIN   0u
/// Biggest value for v-MP core IDs
#define VLL_ID_MP_MAX   7u

/// Core ID for v-SP #0
#define VLL_ID_SP_0     20u
/// Core ID for v-SP #1
#define VLL_ID_SP_1     21u
/// Smallest value for v-SP core IDs
#define VLL_ID_SP_MIN   20u
/// Biggest value for v-SP core IDs
#define VLL_ID_SP_MAX   21u

/// ID for the host processor of the system
#define VLL_ID_HOST     32u

/// Return value indicating successful operation
#define VLL_OK    0
/// Return value indicating error during operation
#define VLL_ERR   -1

/// Return code from EDMA functions indicating free EDMA channel
#define VLL_EDMA_FREE       0u
/// Return code from EDMA functions indicating EDMA channel waiting
#define VLL_EDMA_WAITING    1u
/// Return code from EDMA functions indicating EDMA channel running
#define VLL_EDMA_RUNNING    2u
/// Return code from EDMA functions indicating EDMA channel finished
#define VLL_EDMA_FINISHED   3u
/// Return code from EDMA functions indicating EDMA channel has an error
#define VLL_EDMA_ERROR      4u

/// Return code for mailbox functions indicating successful operation
#define VLL_MBOX_OK    0
/// Return code for mailbox functions indicating error
#define VLL_MBOX_ERR   -1
/// Return code from mailbox functions indicating that functions would block
#define VLL_MBOX_WOULDBLOCK -2
/// Parameter value for mailbox functions indicating CRC checks activated
#define VLL_MBOX_CRC_ENABLE  3
/// Parameter value for mailbox functions indicating CRC checks deactivated
#define VLL_MBOX_CRC_DISABLE 2
/// Parameter value for mailbox functions indicating blocking operation
#define VLL_MBOX_BLOCKING    1
/// Parameter value for mailbox functions indicating non-blocking operation
#define VLL_MBOX_NONBLOCKING 0

/// Mailbox system message payload size in bytes
#define VLL_MSG_PAYLOAD_SIZE     56u

/**
 * @union vid_payload_t
 * @brief Union payload type for mailbox messages
 *
 * @var uchar8 vid_payload_t::pl_ui8
 * Payload of uchar8 data type
 * @var ushort4 vid_payload_t::pl_ui16
 * Payload of ushort4 data type
 * @var uint2 vid_payload_t::pl_ui32
 * Payload of uint2 data type
 * @var ulong1 vid_payload_t::pl_ui64
 * Payload of ulong1 data type
 * @var char8 vid_payload_t::pl_i8
 * Payload of char8 data type
 * @var short4 vid_payload_t::pl_i16
 * Payload of short4 data type
 * @var int2 vid_payload_t::pl_i32
 * Payload of int2 data type
 * @var long1 vid_payload_t::pl_i64
 * Payload of long1 data type
 */
//lint -e9018 Note 9018: declaration of symbol 'unknown-name' with union based type 'const vid_payload_t &' [MISRA 2012 Rule 19.2, advisory]
typedef union {
    uchar8 pl_ui8[VLL_MSG_PAYLOAD_SIZE / 8u];
    ushort4 pl_ui16[VLL_MSG_PAYLOAD_SIZE / 8u];
    uint2 pl_ui32[VLL_MSG_PAYLOAD_SIZE / 8u];
    ulong1 pl_ui64[VLL_MSG_PAYLOAD_SIZE / 8u];
    char8 pl_i8[VLL_MSG_PAYLOAD_SIZE / 8u];
    short4 pl_i16[VLL_MSG_PAYLOAD_SIZE / 8u];
    int2 pl_i32[VLL_MSG_PAYLOAD_SIZE / 8u];
    long1 pl_i64[VLL_MSG_PAYLOAD_SIZE / 8u];
} vid_payload_t;

/**
 * @struct vid_mbox_t
 * @brief Structure mbox type for mailbox messages
 *
 * @var vid_payload_t vid_mbox_t::payload
 * Payload of mailbox message
 * @var int2 vid_mbox_t::type_valid
 * Type (s1) and validity (s0) of mailbox message for functions vid_mbox_send() and vid_mbox_rcv().
 * CRC checksum for functions vid_crc_send() and vid_crc_rcv().
 */

/**
 * @typedef vid_mbox_t
 * @brief Structure mbox type for mailbox messages
 *
 * @var vid_payload_t vid_mbox_t::payload
 * Payload of mailbox message
 * @var int2 vid_mbox_t::type_valid
 * Type (s1) and validity (s0) of mailbox message for functions vid_mbox_send() and vid_mbox_rcv().
 * CRC checksum for functions vid_crc_send() and vid_crc_rcv().
 */
typedef struct vid_mbox_t {
    vid_payload_t payload;
    int2 type_valid;
} vid_mbox_t;

/**
 * @struct vid_hwsema_t
 * @brief Hardware semaphore data type
 *
 * @var uint2 vid_hwsema_t::addr
 * 64-bit address of a hardware semaphore (high part s1 and low part s0)
 * @var uint2 vid_hwsema_t::id
 * ID of a hardware semaphore (both subwords contain the ID)
 */
typedef struct {
    uint2 addr;
    uint2 id;
} vid_hwsema_t;

#endif

//lint -restore
