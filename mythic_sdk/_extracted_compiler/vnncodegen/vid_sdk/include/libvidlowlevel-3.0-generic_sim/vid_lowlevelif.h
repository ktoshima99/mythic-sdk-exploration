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
 * FILENAME: vid_lowlevelif.h
 *
 * DESCRIPTION: videantis low level if API include file for host
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef __VID_LOWLEVELIF_H__
#define __VID_LOWLEVELIF_H__

/**
 * @brief videantis LowLevel Library vid_lowlevelif.h include file
 *
 * @details
 * videantis low level if API include file for host
 *
 * @file vid_lowlevelif.h
 */

#include <stdint.h>
#include "target_consts.h"
#include "target_lowlevel_consts.h"

/// OpenCL-C compatible vector data type uchar8
typedef struct {
    uint8_t s7;
    uint8_t s6;
    uint8_t s5;
    uint8_t s4;
    uint8_t s3;
    uint8_t s2;
    uint8_t s1;
    uint8_t s0;
} uchar8;

/// OpenCL-C compatible vector data type char8
typedef struct {
    int8_t s7;
    int8_t s6;
    int8_t s5;
    int8_t s4;
    int8_t s3;
    int8_t s2;
    int8_t s1;
    int8_t s0;
} char8;

/// OpenCL-C compatible vector data type ushort4
typedef struct {
    uint16_t s3;
    uint16_t s2;
    uint16_t s1;
    uint16_t s0;
} ushort4;

/// OpenCL-C compatible vector data type short4
typedef struct {
    int16_t s3;
    int16_t s2;
    int16_t s1;
    int16_t s0;
} short4;

/// OpenCL-C compatible vector data type uint2
typedef struct {
    uint32_t s1;
    uint32_t s0;
} uint2;

/// OpenCL-C compatible vector data type int2
typedef struct {
    int32_t s1;
    int32_t s0;
} int2;

/// OpenCL-C compatible vector data type float2
typedef struct {
    float s1;
    float s0;
} hfloat2;

/// OpenCL-C compatible vector data type ulong1
typedef struct {
    uint64_t s0;
} ulong1;

/// OpenCL-C compatible vector data type long1
typedef struct {
    int64_t s0;
} long1;

// videantis LowLevel Library common include file
#include "vid_lowlevelif_common.h"


/*
 * Identifiers for HW semaphores
 */
#define VLL_ID_HWSEMA_0  ( 0u)
#define VLL_ID_HWSEMA_1  ( 1u)
#define VLL_ID_HWSEMA_2  ( 2u)
#define VLL_ID_HWSEMA_3  ( 3u)
#define VLL_ID_HWSEMA_4  ( 4u)
#define VLL_ID_HWSEMA_5  ( 5u)
#define VLL_ID_HWSEMA_6  ( 6u)
#define VLL_ID_HWSEMA_7  ( 7u)
#define VLL_ID_HWSEMA_8  ( 8u)
#define VLL_ID_HWSEMA_9  ( 9u)
#define VLL_ID_HWSEMA_10 (10u)
#define VLL_ID_HWSEMA_11 (11u)
#define VLL_ID_HWSEMA_12 (12u)
#define VLL_ID_HWSEMA_13 (13u)
#define VLL_ID_HWSEMA_14 (14u)
#define VLL_ID_HWSEMA_15 (15u)
#define VLL_ID_HWSEMA_16 (16u)
#define VLL_ID_HWSEMA_17 (17u)
#define VLL_ID_HWSEMA_18 (18u)
#define VLL_ID_HWSEMA_19 (19u)

#define VLL_NUM_HWSEMA   (20u)
#define VLL_ID_HWSEMA_MAX (VLL_ID_HWSEMA_19)
#define VLL_ID_HWSEMA_NONE (0xffffffffu)




/*
 * ----------------- API Function prototypes -----------------
 */

/* ------ Class: General Functions ------------ */

// implemented in vid_lowlevel_host.c
int32_t vid_lowlevel_init(void);
int32_t vid_lowlevel_release(void);
int32_t vid_lowlevel_version(uint32_t *major, uint32_t *minor);
int32_t vid_driver_version(uint32_t *major, uint32_t *minor);
void vid_set_verbosity(int level);

// implemented in vid_target_if.c
int32_t vid_mp_core_alloc(uint32_t *core_id);
int32_t vid_sp_core_alloc(uint32_t *core_id);
int32_t vid_core_release(uint32_t core_id);
int32_t vid_boot_core(uint8_t *binptr, uint32_t binsize, uint32_t core_id);
int32_t vid_stop_core(uint32_t core_id);

/* ------ Class: Mailbox System Functions ------------ */

// implemented in vid_mbox_host.c

int32_t vid_mbox_base(uint64_t address);

int32_t vid_mbox_init(uint32_t core_id);
int32_t vid_mbox_send(uint32_t core_id, uint32_t msg_type, vid_payload_t *payload, uint32_t blocking_flag);
int32_t vid_mbox_rcv(uint32_t core_id, uint32_t *msg_type, uint32_t *handle, vid_payload_t *payload, uint32_t blocking_flag);
int32_t vid_mbox_rel(uint32_t core_id, uint32_t handle);

int32_t vid_crc_send(uint32_t core_id, vid_payload_t *payload, uint32_t blocking_flag);
int32_t vid_crc_rcv(uint32_t core_id, uint32_t *handle, vid_payload_t *payload, uint32_t blocking_flag);

int32_t vid_mbox_crc_send(uint32_t core_id,
                            uint32_t msg_type,
                            vid_payload_t *payload,
                            uint32_t blocking_flag,
                            uint32_t crc_mode);
int32_t vid_mbox_crc_rcv(uint32_t core_id,
                           uint32_t *msg_type,
                           uint32_t *handle,
                           vid_payload_t *payload,
                           uint32_t blocking_flag,
                           uint32_t crc_mode);

/* ------ Class: HW semaphore Management Functions ------------ */

// implemented in vid_hwsema_host.c
int32_t vid_hwsema_alloc(vid_hwsema_t *sema);
int32_t vid_hwsema_release(vid_hwsema_t *sema);
int32_t vid_hwsema_getlock(vid_hwsema_t *sema);
int32_t vid_hwsema_rellock(vid_hwsema_t *sema);

/* ------ Class: Memory Management Functions ------------ */

// implemented in vid_memory_host.c
int32_t vid_mem_init(uint64_t *memmap, uint32_t entries);
void * vid_mem_malloc(uint32_t size);
void vid_mem_free(void *ptr);

int32_t vid_ocmem_init(uint64_t *memmap, uint32_t entries);
void * vid_ocmem_malloc(uint32_t size);
void vid_ocmem_free(void *ptr);

uint64_t vid_mem_to_phys(void *ptr);
int32_t vid_mem_is_ovl(void *ptr);

void vid_mem_flush(void);
void vid_mem_inv(void);

int32_t vid_vll_putreg(uint64_t regaddr, uint32_t val);
int32_t vid_vll_getreg(uint64_t regaddr, uint32_t *val);
#endif
