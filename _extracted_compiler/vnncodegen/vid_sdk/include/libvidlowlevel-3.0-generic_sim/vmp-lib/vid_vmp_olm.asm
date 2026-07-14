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
 * FILENAME: vid_vmp_olm.asm
 *
 * DESCRIPTION: videantis v-MP code overlay manager
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP code overlay manager
 *
 * @details
 * This file implements the videantis v-MP code overlay manager.
 * Code overlays are handled automatically by the videantis
 * v-MP code generator vmpasm when the user application jumps
 * to a regular function in a code overlay section.
 *
 * The function VID_VMP_OVERLAY_MANAGER is called from the
 * generated code in vid_vmp_olm_template.asm for each overlay function.
 *
 * @file vid_vmp_olm.asm
 */

/// @cond DOXYGEN_IGNORE_ASM

.equ OLM_ovl_loc_base           = mstmp0  // local IMEM base word address to store the overlay function
.equ OLM_ovl_length             = mstmp1  // length of DMA transfer in number of words
.equ OLM_ovl_ext_base           = mstmp2 // external base address offset from where to load the overlay function

.equ OLM_ovl_v_shift_value      = mvtmp0 // shift value to access group subset of olm.status
.equ OLM_ovl_v_entry_number     = mvtmp1 // entry number of the overlay function inside the overlay group

.equ OLM_ovl_temp_olm           = mstmp3 // scalar tmp register
.equ OLM_ovl_v_mask_group       = mvtmp2 // vector tmp register

// check if memory for olm dsection is defined (default dmem2)
.if !defined(VMP_OLM_MEM)
.equ VMP_OLM_MEM = 2
.endif

// translate olm dsection location
.if {VMP_OLM_MEM == 1}
.equ OLM_MEM = dmem
.elseif {VMP_OLM_MEM == 3}
.equ OLM_MEM = dmem3
.else
.equ OLM_MEM = dmem2
.endif

// global status word of the overlay manager and currently loaded overlays in each section
.dsection olm, OLM_MEM
.org auto

    .alloc status[1] = {0xffffffffffffffff} // all overlays invalid and not loaded
.endsection

.csection vid_vmp_olm_manager
.export VID_VMP_OVERLAY_MANAGER


.equ BITS_PER_GROUP   = {VIDASM_OLM_STATUS_WIDTH / VIDASM_OLM_GROUP_NUMBERS}
.equ MASK_GROUP       = {1<<BITS_PER_GROUP}-1


// the overlay manager entry point, called for each overlay function from vid_vmp_olm_template.asm
//    .type    VID_VMP_OVERLAY_MANAGER, @function
.function VID_VMP_OVERLAY_MANAGER
VID_VMP_OVERLAY_MANAGER:
    // olm values saved in registers OLM_ovl_* in vid_vmp_olm_template.asm

    // setup bit mask to access current overlay group bits from olm.status
.if {BITS_PER_GROUP < 16 }
    V_MVI_64    OLM_ovl_v_mask_group, #MASK_GROUP
.else
    V_MVI_64    OLM_ovl_v_mask_group, #0
    V_MVIL_32   OLM_ovl_v_mask_group, #MASK_GROUP, #0b00001000
.endif

    // align bit mask to position of current overlay group bits from olm.status
    V_SL_U64    OLM_ovl_v_mask_group,   OLM_ovl_v_mask_group,   OLM_ovl_v_shift_value
    // align entry number of requested overlay function to the same position
    V_SL_U64    OLM_ovl_v_entry_number, OLM_ovl_v_entry_number, OLM_ovl_v_shift_value


    // access olm.status and mask other overlay groups
    V_MVFIRI    vidx0, #olm.status
    V_AND_64    OLM_ovl_v_shift_value, (vidx0 @vreg_dmem_dmem2_dmem3), OLM_ovl_v_mask_group
    // check if requested overlay function in this overlay group is already loaded
    V_SUBCS_8   ZERO, OLM_ovl_v_shift_value, OLM_ovl_v_entry_number

    // if already loaded, all subwords zero
    //   jump to overlay function or return to caller in case of *_preload function was triggered
    //   function address in sidx1 was set in vid_olm_*_template.asm
    BVSA_AND_PNT    (sidx1), #COND_Z, #0b11111111

    // if overlay function was not loaded: update olm.status to requested overlay function

    // clear current overlay group bits from olm.status
    MVI         VCONDSEL, #COND_NZ
    V_LOAD      OLM_ovl_v_shift_value, (vidx0 @vreg_dmem_dmem2_dmem3)
    V_MVCR_8    OLM_ovl_v_shift_value, ZERO
    // set new entry number of requested overlay function in current overlay group bits
    V_OR_64     (vidx0 @vreg_dmem_dmem2_dmem3), OLM_ovl_v_shift_value, OLM_ovl_v_entry_number


    // configure & start DMA descriptor for loading overlay function code to instruction memory
    MVFIRI    sidx0, #dma_descr.DMA_CH_OLM_ADDR

    // setup dma_descr.extByteaddr
    // upper 32Bits of external address are constant and same as boot address bits
    MV    STORE_HIGH, BIU_BOOT_ADDRESS_H
    MV    OLM_ovl_temp_olm, BIU_BOOT_ADDRESS_L
    // add lower 32Bits of boot address to external address offset of overlay function
    ADD   (sidx0)+, OLM_ovl_temp_olm, OLM_ovl_ext_base

    // setup dma_descr.xferFlags_intWordaddr_lengthInt_length
    // instruction memory starts at 0xF000
    MVIL    OLM_ovl_temp_olm, 0x0fff
    AND     OLM_ovl_loc_base, OLM_ovl_loc_base, OLM_ovl_temp_olm
    MVIL    OLM_ovl_temp_olm, 0xf000
    ADD     OLM_ovl_temp_olm, OLM_ovl_temp_olm, OLM_ovl_loc_base
    MV      STORE_HIGH, OLM_ovl_temp_olm // intWordaddr
    MV      (sidx0)+, OLM_ovl_length     // length

    // setup dma_descr.reserved_strideInt_count3d_count
    MVI     (sidx0)+, #1 // count

    // trigger DMA read transfer
    .dependency dmem3, 0
    .dependency dmem2, 0
    .dependency dmem, 0
    MVI BIU_DMA_CTRL, #{dma_descr.DMA_CH_OLM_NUM | dma_descr.DMA_READ | dma_descr.DMA_PRIO_STD }

    // wait until DMA transfer is finished
    WAIT #{1<<dma_descr.DMA_CH_OLM_NUM}

    // jump to overlay function or return to caller in case of *_preload function was triggered
    // function address in sidx1 was set in vid_olm_*_template.asm
    JLA    ZERO, (sidx1)

.endfunction
//    .endfunction

.endsection


/// @endcond
