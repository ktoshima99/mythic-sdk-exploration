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
 * FILENAME: vid_vmp_olm_template.asm
 *
 * DESCRIPTION: videantis v-MP overlay manager template file
 *                - sets parameters for VID_VMP_OVERLAY_MANAGER
 *                - VID_VMP_OVERLAY_MANAGER jumps to loaded overlay
 *                - only relative external address offsets to the boot binary
 *                  are supported by this implementation
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// file will be included in its own csection, so it has its own namespace

// general variables provided by assembler:
// VIDASM_LAST_INSTRUCTION_ADDR            (i-th 64-bit address)
// VIDASM_imem_LAST_NON_OVERLAY_ADDRESS (i-th 64-bit address)
//   if 0 == OLM_RELATIVE && 0 == OLM_EXTERNAL_OFFSET: last resident code addr
// VIDASM_imem_LAST_USED_ADDRESS        (i-th 64-bit address)
// VIDASM_OLM_GROUP_NUMBERS             number of overlay regions (default:8)

// overlay manager variables provided by assembler:
// OLM_RELATIVE               (0/1)
// OLM_BOOTADDR_LOCATION      (as passed to assembler by an option, default:0)
// OLM_EXTERNAL_OFFSET        (as passed to assembler by an option, default:0)
// OLM_INTERNAL_START_ADDRESS (instruction address: i-th 64-bit address)
// OLM_EXTERNAL_START_ADDRESS (instruction address: i-th 64-bit address)
// OLM_LENGTH                 (number of 64-bit instructions)
// OLM_GROUP_NUMBER           (group / overlay number)
// OLM_ENTRY_NUMBER           (entry number in group / overlay)
// OLM_INITIAL_STATUS         (status value with initial overlays,
//                             computed according to computations below)

.equ OLM_ovl_loc_base           = mstmp0  // local IMEM base word address to store the overlay function
.equ OLM_ovl_length             = mstmp1  // length of DMA transfer in number of words
.equ OLM_ovl_ext_base           = mstmp2  // external base address offset from where to load the overlay function

.equ OLM_ovl_v_shift_value      = mvtmp0 // shift value to access group subset of vid_vmp_olm.status
.equ OLM_ovl_v_entry_number     = mvtmp1 // entry number of the overlay function inside the overlay group

.equ OLM_ovl_temp_olm           = mstmp3 // scalar tmp register
.equ OLM_ovl_v_mask_group       = mvtmp2 // vector tmp register

// helper constants
.equ BITS_PER_GROUP   = {VIDASM_OLM_STATUS_WIDTH / VIDASM_OLM_GROUP_NUMBERS} // number of bits in vid_vmp_olm.status per overlay group
.equ SHIFT_VALUE      = BITS_PER_GROUP * OLM_GROUP_NUMBER                    // shift value to access group subset of vid_vmp_olm.status
.equ MASK_GROUP       = {1<<BITS_PER_GROUP}-1                                // bit mask to access only bits of a single group


    // save overlay values provided by vmpasm in temp registers
    MVIL   OLM_ovl_ext_base, #{8 * { VIDASM_imem_LAST_NON_OVERLAY_ADDRESS + 1 + OLM_EXTERNAL_START_ADDRESS}}

    MVI    OLM_ovl_loc_base, #OLM_INTERNAL_START_ADDRESS
    MVI    OLM_ovl_length, #OLM_LENGTH

    V_MVI_64    OLM_ovl_v_shift_value, #SHIFT_VALUE
    V_MVI_64    OLM_ovl_v_entry_number, #OLM_ENTRY_NUMBER

    // VID_VMP_OVERLAY_MANAGER jumps to overlay function
    MVFIRI sidx1, #@OLM_ovl_loc_base

    //
    // call overlay-manager
    //
    .fill-slots
    JLR    ZERO, VID_VMP_OVERLAY_MANAGER

.end
// end of file
