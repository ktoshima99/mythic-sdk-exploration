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
 * FILENAME:    vid_generic_if.h
 *
 * DESCRIPTION: Definitions for target-specific lowlevel API functions and data types
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef VID_TARGET_IF_H
#define VID_TARGET_IF_H

/**
 * @brief videantis LowLevel Library vid_generic_if.h include file
 *
 * @details
 * Definitions for target-specific lowlevel API functions and data types
 *
 * @file vid_generic_if.h
 */

#include <stdint.h>

/*
 * --------------- Hardware settings query interface related data types ------------------------
 */

struct vid_freq_t {
    uint32_t  sp_freq_kHz[2u];
    uint32_t  mp_freq_kHz[8u];
    uint32_t  bus_freq_kHz;
};

/*
 * --------------- Target-specific API function prototypes --------------------
 */

/* controlif */

extern int32_t vid_controlif_init(void);
extern int32_t vid_controlif_release(void);
extern void vid_set_verbosity(int32_t level);

extern int32_t vid_boot_core(uint8_t *binptr, uint32_t binsize, uint32_t core_id);
extern int32_t vid_stop_core(uint32_t core_id);
extern int32_t vid_stop_all_cores(void);
extern int32_t vid_reset_target(void);

extern int32_t vid_vll_putreg(uint64_t regaddr, uint32_t val);
extern int32_t vid_vll_getreg(uint64_t regaddr, uint32_t *val);

extern int32_t vid_target_putreg(uint64_t regaddr, uint64_t val);
extern int32_t vid_target_getreg(uint64_t regaddr, uint64_t *val);

extern int32_t vid_mem_upload(uint64_t dst_target, uint8_t *src_host, uint32_t size);
extern int32_t vid_mem_download(uint8_t *dst_host, uint64_t src_target, uint32_t size);

/// @brief calculate temperature from sysmon adc sample
/// equation 2-11 from SYSMON User Guide UG580 (v1.10.1) September 15, 2021
/// For sysmon4e with internal reference and using all 16 bits
/// @return temperature in °C
double vid_target_adc_val_to_temp();

/* hardware settings query interface */

extern void vid_target_get_freq(struct vid_freq_t *freqs);

#endif // defined(VID_TARGET_IF_H)
