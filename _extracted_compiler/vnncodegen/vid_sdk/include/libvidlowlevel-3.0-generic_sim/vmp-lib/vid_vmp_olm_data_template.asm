/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

CONFIDENTIAL AND PROPRIETARY INFORMATION
Copyright 2004 - 2019 videantis GmbH
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
* FILENAME: vid_vmp_olm_data_template.asm
*
* DESCRIPTION: videantis v-MP data overlay manager template file
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

// file will be included in its own csection, so it has its own namespace

// general variables provided by assembler:
// VIDASM_LAST_INSTRUCTION_ADDR	        (i-th 64-bit address)
// VIDASM_imem_LAST_NON_OVERLAY_ADDRESS (i-th 64-bit address)
//   if 0 == OLM_RELATIVE && 0 == OLM_EXTERNAL_OFFSET: last resident code addr
// VIDASM_imem_LAST_USED_ADDRESS        (i-th 64-bit address)
// VIDASM_OLM_DATA_GROUP_NUMBERS        number of data overlay regions(default:8)

// overlay manager variables provided by assembler:
// OLM_DATA_EXTERNAL_OFFSET   (as passed to assembler by an option, default:0)
// OLM_INTERNAL_START_ADDRESS (instruction address: i-th 64-bit address)
// OLM_EXTERNAL_START_ADDRESS (instruction address: i-th 64-bit address)
// OLM_LENGTH                 (number of 64-bit instructions)
// OLM_GROUP_NUMBER           (group / overlay number)
// OLM_ENTRY_NUMBER           (entry number in group / overlay)
// OLM_DATA_INITIAL_STATUS    (status value with initial overlays,
//                             computed according to computations below)

	// register usage in code
	.equ INTERNAL_START = R0
	.equ EXTERNAL_START = R1
	.equ LENGTH = R2
	.equ MASK = R4
	.equ DEBUG = 0
	
	// helper variables
	.equ EXTERNAL_START_IN_BYTES = OLM_DATA_EXTERNAL_OFFSET + { OLM_EXTERNAL_START_ADDRESS * 4 }
	.equ MASK_SHIFT_VALUE = { 8 * OLM_GROUP_NUMBER } + OLM_ENTRY_NUMBER

	//
	// code to call overlay-manager
	//
read:	
	// simply return
	//JLA	ZERO, R31

write:	
	// simply return
	//JLA	ZERO, R31

	.end
// end of file
