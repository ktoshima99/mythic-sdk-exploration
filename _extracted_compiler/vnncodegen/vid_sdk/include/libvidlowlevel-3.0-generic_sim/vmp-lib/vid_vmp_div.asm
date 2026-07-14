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
 * FILENAME: vid_vmp_div.asm
 *
 * DESCRIPTION: videantis v-MP integer division functions
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP integer division functions
 *
 * @details
 * This file is the compiled version of vid_vmp_div.cl and
 * implements the videantis v-MP integer division functions
 * vid_vmp_libmath@955f40a6a04a05684a454c34e28bc928a430847f
 *
 * @file vid_vmp_div.asm
 */

/// @cond DOXYGEN_IGNORE_ASM

// start of module
// compile time: 2025-Feb-18 10:01:30
// cpu:mp4.1
// level of optimization: -O3
	.option "enable-cross-pipeline-memory-delay"
	.option "enable-memory-def-use-delay"

	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.export	__udivmodsi4                    // -- Begin function _udivmodsi4
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__udivmodsi4
	.org auto
//	.type	__udivmodsi4, @function
__udivmodsi4:                           // @_udivmodsi4
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	SUBCS	SZERO, SR0, SR1
	MVI	SR3, #0x0
	BSSR	LBB0_2, #0 /* C/Lu */
// %bb.1:
	MV	VR0, SR1
	V_MIXR_32	VR0, VR0, VR0
	V_SUB_32	VR1, VZERO, VR0
	V_CMBI_32	VR2, VR0, #0x0
	V_MVI_32	VR3, #0x1
	V_SL_32	VR3, VR3, VR2
	V_SL_32	VR2, VR1, VR2
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_ADD_32	VR2, VR3, VR2
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR1, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR1
	MV	VR1, SR0
	V_MIXL_32	VR3, VR62, VR63
	V_MIXR_32	VR1, VR1, VR1
	V_ADD_32	VR2, VR2, VR3
	V_MACPL0_U32	VR62, VR1, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_MUL_32	VR3, VR2, VR0
	V_SUB_32	VR1, VR1, VR3
	V_MVI_32	VR3, #0x0
	V_SUBCS_32	VZERO, VR1, VR0
	MVI	VCONDSEL, #0xb
	V_MVI_32	VR4, #0x0
	V_MVCR_32	VR4, VR0
	V_SUB_32	VR1, VR1, VR4
	V_MV_32	VR4, VZERO
	V_ADDICR_32	VR4, VZERO, #0x1
	V_SUBCS_32	VZERO, VR1, VR0
	V_ADD_32	VR2, VR2, VR4
	V_MV_32	VR4, VZERO
	V_ADDICR_32	VR4, VZERO, #0x1
	V_MVCR_32	VR3, VR0
	V_ADD_32	VR0, VR2, VR4
	V_SUB_32	VR1, VR1, VR3
	MV	SR0, VR1
	MV	SR3, VR0
LBB0_2:
	SRI_U	SR1, SR2, #0x3
	MV	SFIR0, SR1
	MV	(SFIR0), SR0
	MV	SR0, SR3
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __udivmodsi4

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__udivmod_32                    // -- Begin function _udivmod_32
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__udivmod_32
	.org auto
//	.type	__udivmod_32, @function
__udivmod_32:                           // @_udivmod_32
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_32	VR2, VZERO, VR1
	V_CMBI_32	VR3, VR1, #0x0
	V_MVI_32	VR4, #0x1
	V_SL_32	VR4, VR4, VR3
	V_SL_32	VR3, VR2, VR3
	V_MACPL0_U32	VR62, VR4, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR3, VR4, VR3
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR2, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_ADD_32	VR2, VR3, VR2
	V_MACPL0_U32	VR62, VR0, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_MUL_32	VR3, VR2, VR1
	V_MVI_32	VR4, #0x0
	V_SUB_32	VR0, VR0, VR3
	V_SUBCS_32	VZERO, VR0, VR1
	MVI	VCONDSEL, #0xb
	V_MVI_32	VR3, #0x0
	V_MVCR_32	VR3, VR1
	V_SUB_32	VR3, VR0, VR3
	V_MV_32	VR0, VZERO
	V_ADDICR_32	VR0, VZERO, #0x1
	V_SUBCS_32	VZERO, VR3, VR1
	SRI_U	SR0, SR0, #0x3
	V_ADD_32	VR0, VR2, VR0
	V_MV_32	VR2, VZERO
	V_ADDICR_32	VR2, VZERO, #0x1
	V_ADD_32	VR0, VR0, VR2
	V_MVCR_32	VR4, VR1
	MV	VFIR0, SR0
	V_SUB_32	(VFIR0), VR3, VR4
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __udivmod_32

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__umodsi3                       // -- Begin function _umodsi3
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__umodsi3
	.org auto
//	.type	__umodsi3, @function
__umodsi3:                              // @_umodsi3
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	SUBCS	SZERO, SR0, SR1
	BSSR	LBB2_2, #0 /* C/Lu */
// %bb.1:
	MV	VR0, SR1
	V_MIXR_32	VR0, VR0, VR0
	V_SUB_32	VR1, VZERO, VR0
	V_CMBI_32	VR2, VR0, #0x0
	V_MVI_32	VR3, #0x1
	V_SL_32	VR3, VR3, VR2
	V_SL_32	VR2, VR1, VR2
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_ADD_32	VR2, VR3, VR2
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR1, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR1
	MV	VR1, SR0
	V_MIXL_32	VR3, VR62, VR63
	V_MIXR_32	VR1, VR1, VR1
	V_ADD_32	VR2, VR2, VR3
	V_MACPL0_U32	VR62, VR1, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_MUL_32	VR2, VR2, VR0
	V_SUB_32	VR1, VR1, VR2
	V_SUBCS_32	VZERO, VR1, VR0
	V_MVI_32	VR2, #0x0
	MVI	VCONDSEL, #0x0
	V_MV_64	VR3, VR0
	V_MVCR_32	VR3, VR2
	V_SUB_32	VR1, VR1, VR3
	V_SUBCS_32	VZERO, VR1, VR0
	V_MVCR_32	VR0, VR2
	V_SUB_32	VR0, VR1, VR0
	MV	SR0, VR0
LBB2_2:
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __umodsi3

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__umod_32                       // -- Begin function _umod_32
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__umod_32
	.org auto
//	.type	__umod_32, @function
__umod_32:                              // @_umod_32
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_32	VR2, VZERO, VR1
	V_CMBI_32	VR3, VR1, #0x0
	V_MVI_32	VR4, #0x1
	V_SL_32	VR4, VR4, VR3
	V_SL_32	VR3, VR2, VR3
	V_MACPL0_U32	VR62, VR4, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR3, VR4, VR3
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR2, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_ADD_32	VR2, VR3, VR2
	V_MACPL0_U32	VR62, VR0, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_MUL_32	VR2, VR2, VR1
	V_SUB_32	VR0, VR0, VR2
	V_MVI_32	VR2, #0x0
	V_SUBCS_32	VZERO, VR0, VR1
	MVI	VCONDSEL, #0x0
	V_MV_64	VR3, VR1
	V_MVCR_32	VR3, VR2
	V_SUB_32	VR0, VR0, VR3
	V_SUBCS_32	VZERO, VR0, VR1
	V_MVCR_32	VR1, VR2
	V_SUB_32	VR0, VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __umod_32

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__udivsi3                       // -- Begin function _udivsi3
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__udivsi3
	.org auto
//	.type	__udivsi3, @function
__udivsi3:                              // @_udivsi3
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	SUBCS	SZERO, SR0, SR1
	MVI	SR2, #0x0
	BSSR	LBB4_2, #0 /* C/Lu */
// %bb.1:
	MV	VR0, SR1
	V_MIXR_32	VR0, VR0, VR0
	V_SUB_32	VR1, VZERO, VR0
	V_CMBI_32	VR2, VR0, #0x0
	V_MVI_32	VR3, #0x1
	V_SL_32	VR3, VR3, VR2
	V_SL_32	VR2, VR1, VR2
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_ADD_32	VR2, VR3, VR2
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR1, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR1
	MV	VR1, SR0
	V_MIXL_32	VR3, VR62, VR63
	V_MIXR_32	VR1, VR1, VR1
	V_ADD_32	VR2, VR2, VR3
	V_MACPL0_U32	VR62, VR1, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_MUL_32	VR3, VR2, VR0
	V_SUB_32	VR1, VR1, VR3
	V_MVI_32	VR3, #0x0
	V_SUBCS_32	VZERO, VR1, VR0
	MVI	VCONDSEL, #0xb
	V_MVCR_32	VR3, VR0
	V_SUB_32	VR1, VR1, VR3
	V_MV_32	VR3, VZERO
	V_ADDICR_32	VR3, VZERO, #0x1
	V_SUBCS_32	VZERO, VR1, VR0
	V_ADD_32	VR0, VR2, VR3
	V_MV_32	VR1, VZERO
	V_ADDICR_32	VR1, VZERO, #0x1
	V_ADD_32	VR0, VR0, VR1
	MV	SR2, VR0
LBB4_2:
	MV	SR0, SR2
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __udivsi3

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__udiv_32                       // -- Begin function _udiv_32
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__udiv_32
	.org auto
//	.type	__udiv_32, @function
__udiv_32:                              // @_udiv_32
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_32	VR2, VZERO, VR1
	V_CMBI_32	VR3, VR1, #0x0
	V_MVI_32	VR4, #0x1
	V_SL_32	VR4, VR4, VR3
	V_SL_32	VR3, VR2, VR3
	V_MACPL0_U32	VR62, VR4, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR3, VR4, VR3
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR2, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_ADD_32	VR2, VR3, VR2
	V_MACPL0_U32	VR62, VR0, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_MUL_32	VR3, VR2, VR1
	V_SUB_32	VR0, VR0, VR3
	V_MVI_32	VR3, #0x0
	V_SUBCS_32	VZERO, VR0, VR1
	MVI	VCONDSEL, #0xb
	V_MVCR_32	VR3, VR1
	V_SUB_32	VR0, VR0, VR3
	V_MV_32	VR3, VZERO
	V_ADDICR_32	VR3, VZERO, #0x1
	V_SUBCS_32	VZERO, VR0, VR1
	V_ADD_32	VR0, VR2, VR3
	V_MV_32	VR1, VZERO
	V_ADDICR_32	VR1, VZERO, #0x1
	V_ADD_32	VR0, VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __udiv_32

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__divmodsi4                     // -- Begin function _divmodsi4
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__divmodsi4
	.org auto
//	.type	__divmodsi4, @function
__divmodsi4:                            // @_divmodsi4
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	MV	VR0, SR1
	V_MIXR_32	VR0, VR0, VR0
	V_ABSADDI_32	VR1, VR0, #0x0
	V_SUB_32	VR2, VZERO, VR1
	V_CMBI_32	VR3, VR1, #0x0
	V_MVI_32	VR4, #0x1
	V_SL_32	VR4, VR4, VR3
	V_SL_32	VR3, VR2, VR3
	V_MACPL0_U32	VR62, VR4, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR3, VR4, VR3
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR2, VR3, VR2
	MV	VR4, SR0
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXR_32	VR2, VR4, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ABSADDI_32	VR5, VR2, #0x0
	V_ADD_32	VR3, VR3, VR4
	V_MACPL0_U32	VR62, VR5, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_MUL_32	VR4, VR3, VR1
	V_SUB_32	VR4, VR5, VR4
	V_MVI_32	VR5, #0x0
	V_SUBCS_32	VZERO, VR4, VR1
	MVI	VCONDSEL, #0xb
	V_MVI_32	VR58, #0x0
	V_MVCR_32	VR58, VR1
	V_SUB_32	VR4, VR4, VR58
	V_MV_32	VR58, VZERO
	V_ADDICR_32	VR58, VZERO, #0x1
	V_SUBCS_32	VZERO, VR4, VR1
	V_ADD_32	VR3, VR3, VR58
	V_MV_32	VR58, VZERO
	V_ADDICR_32	VR58, VZERO, #0x1
	V_XOR_32	VR0, VR0, VR2
	V_ADD_32	VR3, VR3, VR58
	V_MVCR_32	VR5, VR1
	V_SUB_32	VR1, VR4, VR5
	V_SUB_32	VR4, VZERO, VR3
	V_SUBICS_32	VZERO, VR0, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_32	VR3, VR4
	V_SUB_32	VR0, VZERO, VR1
	V_SUBICS_32	VZERO, VR2, #0x0
	SRI_U	SR0, SR2, #0x3
	V_MVCR_32	VR1, VR0
	MV	SR1, VR1
	MV	SFIR0, SR0
	MV	SR0, VR3
	MV	(SFIR0), SR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __divmodsi4

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__divmod_32                     // -- Begin function _divmod_32
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__divmod_32
	.org auto
//	.type	__divmod_32, @function
__divmod_32:                            // @_divmod_32
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_32	VR2, VR1, #0x0
	V_SUB_32	VR3, VZERO, VR2
	V_CMBI_32	VR4, VR2, #0x0
	V_MVI_32	VR5, #0x1
	V_SL_32	VR5, VR5, VR4
	V_SL_32	VR4, VR3, VR4
	V_MACPL0_U32	VR62, VR5, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR4, VR5, VR4
	V_MUL_32	VR5, VR4, VR3
	V_MACPL0_U32	VR62, VR4, VR5
	V_MIXL_32	VR5, VR62, VR63
	V_ADD_32	VR4, VR4, VR5
	V_MUL_32	VR5, VR4, VR3
	V_MACPL0_U32	VR62, VR4, VR5
	V_MIXL_32	VR5, VR62, VR63
	V_ADD_32	VR4, VR4, VR5
	V_MUL_32	VR5, VR4, VR3
	V_MACPL0_U32	VR62, VR4, VR5
	V_MIXL_32	VR5, VR62, VR63
	V_ADD_32	VR4, VR4, VR5
	V_MUL_32	VR3, VR4, VR3
	V_MACPL0_U32	VR62, VR4, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ABSADDI_32	VR5, VR0, #0x0
	V_ADD_32	VR3, VR4, VR3
	V_MACPL0_U32	VR62, VR5, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_MUL_32	VR4, VR3, VR2
	V_MVI_32	VR58, #0x0
	V_SUB_32	VR4, VR5, VR4
	V_SUBCS_32	VZERO, VR4, VR2
	MVI	VCONDSEL, #0xb
	V_MVI_32	VR5, #0x0
	V_MVCR_32	VR5, VR2
	V_SUB_32	VR4, VR4, VR5
	V_MV_32	VR5, VZERO
	V_ADDICR_32	VR5, VZERO, #0x1
	V_SUBCS_32	VZERO, VR4, VR2
	V_ADD_32	VR3, VR3, VR5
	V_MV_32	VR5, VZERO
	V_ADDICR_32	VR5, VZERO, #0x1
	V_XOR_32	VR59, VR1, VR0
	V_ADD_32	VR1, VR3, VR5
	V_MVCR_32	VR58, VR2
	V_SUB_32	VR2, VR4, VR58
	V_SUB_32	VR3, VZERO, VR1
	V_SUBICS_32	VZERO, VR59, #0x0
	MVI	VCONDSEL, #0x3
	SRI_U	SR0, SR0, #0x3
	V_MVCR_32	VR1, VR3
	V_SUB_32	VR3, VZERO, VR2
	V_SUBICS_32	VZERO, VR0, #0x0
	V_MVCR_32	VR2, VR3
	MV	VFIR0, SR0
	V_MV_32	(VFIR0), VR2
	V_MV_64	VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __divmod_32

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__modsi3                        // -- Begin function _modsi3
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__modsi3
	.org auto
//	.type	__modsi3, @function
__modsi3:                               // @_modsi3
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	MV	VR0, SR1
	V_MIXR_32	VR0, VR0, VR0
	V_ABSADDI_32	VR0, VR0, #0x0
	V_SUB_32	VR1, VZERO, VR0
	V_CMBI_32	VR2, VR0, #0x0
	V_MVI_32	VR3, #0x1
	V_SL_32	VR3, VR3, VR2
	V_SL_32	VR2, VR1, VR2
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_ADD_32	VR2, VR3, VR2
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR3, VR2, VR1
	V_MACPL0_U32	VR62, VR2, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR2, VR2, VR3
	V_MUL_32	VR1, VR2, VR1
	MV	VR3, SR0
	V_MACPL0_U32	VR62, VR2, VR1
	V_MIXR_32	VR1, VR3, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ABSADDI_32	VR4, VR1, #0x0
	V_ADD_32	VR2, VR2, VR3
	V_MACPL0_U32	VR62, VR4, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_MUL_32	VR2, VR2, VR0
	V_SUB_32	VR2, VR4, VR2
	V_SUBCS_32	VZERO, VR2, VR0
	V_MVI_32	VR3, #0x0
	MVI	VCONDSEL, #0x0
	V_MV_64	VR4, VR0
	V_MVCR_32	VR4, VR3
	V_SUB_32	VR2, VR2, VR4
	V_SUBCS_32	VZERO, VR2, VR0
	V_MVCR_32	VR0, VR3
	V_SUB_32	VR0, VR2, VR0
	V_SUB_32	VR2, VZERO, VR0
	V_SUBICS_32	VZERO, VR1, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_32	VR0, VR2
	MV	SR0, VR0
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __modsi3

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__mod_32                        // -- Begin function _mod_32
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__mod_32
	.org auto
//	.type	__mod_32, @function
__mod_32:                               // @_mod_32
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_32	VR1, VR1, #0x0
	V_SUB_32	VR2, VZERO, VR1
	V_CMBI_32	VR3, VR1, #0x0
	V_MVI_32	VR4, #0x1
	V_SL_32	VR4, VR4, VR3
	V_SL_32	VR3, VR2, VR3
	V_MACPL0_U32	VR62, VR4, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR3, VR4, VR3
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR2, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_ABSADDI_32	VR4, VR0, #0x0
	V_ADD_32	VR2, VR3, VR2
	V_MACPL0_U32	VR62, VR4, VR2
	V_MIXL_32	VR2, VR62, VR63
	V_MUL_32	VR2, VR2, VR1
	V_SUB_32	VR2, VR4, VR2
	V_MVI_32	VR3, #0x0
	V_SUBCS_32	VZERO, VR2, VR1
	MVI	VCONDSEL, #0x0
	V_MV_64	VR4, VR1
	V_MVCR_32	VR4, VR3
	V_SUB_32	VR2, VR2, VR4
	V_SUBCS_32	VZERO, VR2, VR1
	V_MVCR_32	VR1, VR3
	V_SUB_32	VR1, VR2, VR1
	V_SUB_32	VR2, VZERO, VR1
	V_SUBICS_32	VZERO, VR0, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_32	VR1, VR2
	V_MV_64	VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __mod_32

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__divsi3                        // -- Begin function _divsi3
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__divsi3
	.org auto
//	.type	__divsi3, @function
__divsi3:                               // @_divsi3
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	MV	VR0, SR1
	V_MIXR_32	VR0, VR0, VR0
	V_ABSADDI_32	VR1, VR0, #0x0
	V_SUB_32	VR2, VZERO, VR1
	V_CMBI_32	VR3, VR1, #0x0
	V_MVI_32	VR4, #0x1
	V_SL_32	VR4, VR4, VR3
	V_SL_32	VR3, VR2, VR3
	V_MACPL0_U32	VR62, VR4, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ADD_32	VR3, VR4, VR3
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR4, VR3, VR2
	V_MACPL0_U32	VR62, VR3, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR3, VR3, VR4
	V_MUL_32	VR2, VR3, VR2
	MV	VR4, SR0
	V_MACPL0_U32	VR62, VR3, VR2
	V_MIXR_32	VR2, VR4, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ABSADDI_32	VR5, VR2, #0x0
	V_ADD_32	VR3, VR3, VR4
	V_MACPL0_U32	VR62, VR5, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_MUL_32	VR4, VR3, VR1
	V_SUB_32	VR4, VR5, VR4
	V_SUBCS_32	VZERO, VR4, VR1
	MVI	VCONDSEL, #0xb
	V_MVI_32	VR5, #0x0
	V_MVCR_32	VR5, VR1
	V_SUB_32	VR4, VR4, VR5
	V_MV_32	VR5, VZERO
	V_ADDICR_32	VR5, VZERO, #0x1
	V_SUBCS_32	VZERO, VR4, VR1
	V_ADD_32	VR1, VR3, VR5
	V_MV_32	VR3, VZERO
	V_ADDICR_32	VR3, VZERO, #0x1
	V_XOR_32	VR0, VR0, VR2
	V_ADD_32	VR1, VR1, VR3
	V_SUB_32	VR2, VZERO, VR1
	V_SUBICS_32	VZERO, VR0, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_32	VR1, VR2
	MV	SR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __divsi3

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__div_32                        // -- Begin function _div_32
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__div_32
	.org auto
//	.type	__div_32, @function
__div_32:                               // @_div_32
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_32	VR2, VR1, #0x0
	V_SUB_32	VR3, VZERO, VR2
	V_CMBI_32	VR4, VR2, #0x0
	V_MVI_32	VR5, #0x1
	V_SL_32	VR5, VR5, VR4
	V_SL_32	VR4, VR3, VR4
	V_MACPL0_U32	VR62, VR5, VR4
	V_MIXL_32	VR4, VR62, VR63
	V_ADD_32	VR4, VR5, VR4
	V_MUL_32	VR5, VR4, VR3
	V_MACPL0_U32	VR62, VR4, VR5
	V_MIXL_32	VR5, VR62, VR63
	V_ADD_32	VR4, VR4, VR5
	V_MUL_32	VR5, VR4, VR3
	V_MACPL0_U32	VR62, VR4, VR5
	V_MIXL_32	VR5, VR62, VR63
	V_ADD_32	VR4, VR4, VR5
	V_MUL_32	VR5, VR4, VR3
	V_MACPL0_U32	VR62, VR4, VR5
	V_MIXL_32	VR5, VR62, VR63
	V_ADD_32	VR4, VR4, VR5
	V_MUL_32	VR3, VR4, VR3
	V_MACPL0_U32	VR62, VR4, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_ABSADDI_32	VR5, VR0, #0x0
	V_ADD_32	VR3, VR4, VR3
	V_MACPL0_U32	VR62, VR5, VR3
	V_MIXL_32	VR3, VR62, VR63
	V_MUL_32	VR4, VR3, VR2
	V_SUB_32	VR4, VR5, VR4
	V_MVI_32	VR5, #0x0
	V_SUBCS_32	VZERO, VR4, VR2
	MVI	VCONDSEL, #0xb
	V_MVCR_32	VR5, VR2
	V_SUB_32	VR4, VR4, VR5
	V_MV_32	VR5, VZERO
	V_ADDICR_32	VR5, VZERO, #0x1
	V_SUBCS_32	VZERO, VR4, VR2
	V_ADD_32	VR2, VR3, VR5
	V_MV_32	VR3, VZERO
	V_ADDICR_32	VR3, VZERO, #0x1
	V_XOR_32	VR1, VR1, VR0
	V_ADD_32	VR0, VR2, VR3
	V_SUB_32	VR2, VZERO, VR0
	V_SUBICS_32	VZERO, VR1, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_32	VR0, VR2
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __div_32

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__udivmod_16                    // -- Begin function _udivmod_16
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__udivmod_16
	.org auto
//	.type	__udivmod_16, @function
__udivmod_16:                           // @_udivmod_16
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_16	VR2, VZERO, VR1
	V_CMBI_16	VR3, VR1, #0x0
	V_MVI_16	VR4, #0x1
	V_SL_16	VR4, VR4, VR3
	V_SL_16	VR3, VR2, VR3
	V_MACPL0_U16	VR62, VR4, VR3
	MVIL	VPERMREG, #0xb090301
	V_PERMREG_16	VR3, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR4, VR3
	V_MUL_16	VR4, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR4
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR3, VR4
	V_MUL_16	VR4, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR4
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR3, VR4
	V_MUL_16	VR2, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR2
	V_PERMREG_16	VR2, VR63, VR62, VPERMREG
	V_ADD_16	VR2, VR3, VR2
	V_MACPL0_U16	VR62, VR0, VR2
	V_PERMREG_16	VR2, VR63, VR62, VPERMREG
	V_MUL_16	VR3, VR2, VR1
	V_MVI_16	VR4, #0x0
	V_SUB_16	VR0, VR0, VR3
	V_SUBCS_16	VZERO, VR0, VR1
	MVI	VCONDSEL, #0xb
	V_MVI_16	VR3, #0x0
	V_MVCR_16	VR3, VR1
	V_SUB_16	VR3, VR0, VR3
	V_MV_16	VR0, VZERO
	V_ADDICR_16	VR0, VZERO, #0x1
	V_SUBCS_16	VZERO, VR3, VR1
	SRI_U	SR0, SR0, #0x3
	V_ADD_16	VR0, VR2, VR0
	V_MV_16	VR2, VZERO
	V_ADDICR_16	VR2, VZERO, #0x1
	V_ADD_16	VR0, VR0, VR2
	V_MVCR_16	VR4, VR1
	MV	VFIR0, SR0
	V_SUB_16	(VFIR0), VR3, VR4
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __udivmod_16

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__umod_16                       // -- Begin function _umod_16
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__umod_16
	.org auto
//	.type	__umod_16, @function
__umod_16:                              // @_umod_16
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_16	VR2, VZERO, VR1
	V_CMBI_16	VR3, VR1, #0x0
	V_MVI_16	VR4, #0x1
	V_SL_16	VR4, VR4, VR3
	V_SL_16	VR3, VR2, VR3
	V_MACPL0_U16	VR62, VR4, VR3
	MVIL	VPERMREG, #0xb090301
	V_PERMREG_16	VR3, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR4, VR3
	V_MUL_16	VR4, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR4
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR3, VR4
	V_MUL_16	VR4, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR4
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR3, VR4
	V_MUL_16	VR2, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR2
	V_PERMREG_16	VR2, VR63, VR62, VPERMREG
	V_ADD_16	VR2, VR3, VR2
	V_MACPL0_U16	VR62, VR0, VR2
	V_PERMREG_16	VR2, VR63, VR62, VPERMREG
	V_MUL_16	VR2, VR2, VR1
	V_SUB_16	VR0, VR0, VR2
	V_MVI_16	VR2, #0x0
	V_SUBCS_16	VZERO, VR0, VR1
	MVI	VCONDSEL, #0x0
	V_MV_64	VR3, VR1
	V_MVCR_16	VR3, VR2
	V_SUB_16	VR0, VR0, VR3
	V_SUBCS_16	VZERO, VR0, VR1
	V_MVCR_16	VR1, VR2
	V_SUB_16	VR0, VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __umod_16

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__udiv_16                       // -- Begin function _udiv_16
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__udiv_16
	.org auto
//	.type	__udiv_16, @function
__udiv_16:                              // @_udiv_16
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_16	VR2, VZERO, VR1
	V_CMBI_16	VR3, VR1, #0x0
	V_MVI_16	VR4, #0x1
	V_SL_16	VR4, VR4, VR3
	V_SL_16	VR3, VR2, VR3
	V_MACPL0_U16	VR62, VR4, VR3
	MVIL	VPERMREG, #0xb090301
	V_PERMREG_16	VR3, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR4, VR3
	V_MUL_16	VR4, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR4
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR3, VR4
	V_MUL_16	VR4, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR4
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR3, VR4
	V_MUL_16	VR2, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR2
	V_PERMREG_16	VR2, VR63, VR62, VPERMREG
	V_ADD_16	VR2, VR3, VR2
	V_MACPL0_U16	VR62, VR0, VR2
	V_PERMREG_16	VR2, VR63, VR62, VPERMREG
	V_MUL_16	VR3, VR2, VR1
	V_SUB_16	VR0, VR0, VR3
	V_MVI_16	VR3, #0x0
	V_SUBCS_16	VZERO, VR0, VR1
	MVI	VCONDSEL, #0xb
	V_MVCR_16	VR3, VR1
	V_SUB_16	VR0, VR0, VR3
	V_MV_16	VR3, VZERO
	V_ADDICR_16	VR3, VZERO, #0x1
	V_SUBCS_16	VZERO, VR0, VR1
	V_ADD_16	VR0, VR2, VR3
	V_MV_16	VR1, VZERO
	V_ADDICR_16	VR1, VZERO, #0x1
	V_ADD_16	VR0, VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __udiv_16

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__udivmod_8                     // -- Begin function _udivmod_8
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__udivmod_8
	.org auto
//	.type	__udivmod_8, @function
__udivmod_8:                            // @_udivmod_8
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_8	VR2, VZERO, VR1
	V_CMBI_8	VR3, VR1, #0x0
	V_MVI_8	VR4, #0x1
	V_SL_8	VR4, VR4, VR3
	V_SL_8	VR3, VR2, VR3
	V_MACPL0_U8	VR62, VR4, VR3
	MVIL	VPERMREG, #-0x2468acf
	V_PERMREG_8	VR3, VR63, VR62, VPERMREG
	V_ADD_8	VR3, VR4, VR3
	V_MUL_8	VR4, VR3, VR2
	V_MACPL0_U8	VR62, VR3, VR4
	V_PERMREG_8	VR4, VR63, VR62, VPERMREG
	V_ADD_8	VR3, VR3, VR4
	V_MUL_8	VR2, VR3, VR2
	V_MACPL0_U8	VR62, VR3, VR2
	V_PERMREG_8	VR2, VR63, VR62, VPERMREG
	V_ADD_8	VR2, VR3, VR2
	V_MACPL0_U8	VR62, VR0, VR2
	V_PERMREG_8	VR2, VR63, VR62, VPERMREG
	V_MUL_8	VR3, VR2, VR1
	V_MVI_8	VR4, #0x0
	V_SUB_8	VR0, VR0, VR3
	V_SUBCS_8	VZERO, VR0, VR1
	MVI	VCONDSEL, #0xb
	V_MVI_8	VR3, #0x0
	V_MVCR_8	VR3, VR1
	V_SUB_8	VR3, VR0, VR3
	V_MV_8	VR0, VZERO
	V_ADDICR_8	VR0, VZERO, #0x1
	V_SUBCS_8	VZERO, VR3, VR1
	SRI_U	SR0, SR0, #0x3
	V_ADD_8	VR0, VR2, VR0
	V_MV_8	VR2, VZERO
	V_ADDICR_8	VR2, VZERO, #0x1
	V_ADD_8	VR0, VR0, VR2
	V_MVCR_8	VR4, VR1
	MV	VFIR0, SR0
	V_SUB_8	(VFIR0), VR3, VR4
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __udivmod_8

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__umod_8                        // -- Begin function _umod_8
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__umod_8
	.org auto
//	.type	__umod_8, @function
__umod_8:                               // @_umod_8
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_8	VR2, VZERO, VR1
	V_CMBI_8	VR3, VR1, #0x0
	V_MVI_8	VR4, #0x1
	V_SL_8	VR4, VR4, VR3
	V_SL_8	VR3, VR2, VR3
	V_MACPL0_U8	VR62, VR4, VR3
	MVIL	VPERMREG, #-0x2468acf
	V_PERMREG_8	VR3, VR63, VR62, VPERMREG
	V_ADD_8	VR3, VR4, VR3
	V_MUL_8	VR4, VR3, VR2
	V_MACPL0_U8	VR62, VR3, VR4
	V_PERMREG_8	VR4, VR63, VR62, VPERMREG
	V_ADD_8	VR3, VR3, VR4
	V_MUL_8	VR2, VR3, VR2
	V_MACPL0_U8	VR62, VR3, VR2
	V_PERMREG_8	VR2, VR63, VR62, VPERMREG
	V_ADD_8	VR2, VR3, VR2
	V_MACPL0_U8	VR62, VR0, VR2
	V_PERMREG_8	VR2, VR63, VR62, VPERMREG
	V_MUL_8	VR2, VR2, VR1
	V_SUB_8	VR0, VR0, VR2
	V_MVI_8	VR2, #0x0
	V_SUBCS_8	VZERO, VR0, VR1
	MVI	VCONDSEL, #0x0
	V_MV_64	VR3, VR1
	V_MVCR_8	VR3, VR2
	V_SUB_8	VR0, VR0, VR3
	V_SUBCS_8	VZERO, VR0, VR1
	V_MVCR_8	VR1, VR2
	V_SUB_8	VR0, VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __umod_8

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__udiv_8                        // -- Begin function _udiv_8
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__udiv_8
	.org auto
//	.type	__udiv_8, @function
__udiv_8:                               // @_udiv_8
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_SUB_8	VR2, VZERO, VR1
	V_CMBI_8	VR3, VR1, #0x0
	V_MVI_8	VR4, #0x1
	V_SL_8	VR4, VR4, VR3
	V_SL_8	VR3, VR2, VR3
	V_MACPL0_U8	VR62, VR4, VR3
	MVIL	VPERMREG, #-0x2468acf
	V_PERMREG_8	VR3, VR63, VR62, VPERMREG
	V_ADD_8	VR3, VR4, VR3
	V_MUL_8	VR4, VR3, VR2
	V_MACPL0_U8	VR62, VR3, VR4
	V_PERMREG_8	VR4, VR63, VR62, VPERMREG
	V_ADD_8	VR3, VR3, VR4
	V_MUL_8	VR2, VR3, VR2
	V_MACPL0_U8	VR62, VR3, VR2
	V_PERMREG_8	VR2, VR63, VR62, VPERMREG
	V_ADD_8	VR2, VR3, VR2
	V_MACPL0_U8	VR62, VR0, VR2
	V_PERMREG_8	VR2, VR63, VR62, VPERMREG
	V_MUL_8	VR3, VR2, VR1
	V_SUB_8	VR0, VR0, VR3
	V_MVI_8	VR3, #0x0
	V_SUBCS_8	VZERO, VR0, VR1
	MVI	VCONDSEL, #0xb
	V_MVCR_8	VR3, VR1
	V_SUB_8	VR0, VR0, VR3
	V_MV_8	VR3, VZERO
	V_ADDICR_8	VR3, VZERO, #0x1
	V_SUBCS_8	VZERO, VR0, VR1
	V_ADD_8	VR0, VR2, VR3
	V_MV_8	VR1, VZERO
	V_ADDICR_8	VR1, VZERO, #0x1
	V_ADD_8	VR0, VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __udiv_8

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__divmod_16                     // -- Begin function _divmod_16
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__divmod_16
	.org auto
//	.type	__divmod_16, @function
__divmod_16:                            // @_divmod_16
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_16	VR2, VR1, #0x0
	V_SUB_16	VR3, VZERO, VR2
	V_CMBI_16	VR4, VR2, #0x0
	V_MVI_16	VR5, #0x1
	V_SL_16	VR5, VR5, VR4
	V_SL_16	VR4, VR3, VR4
	V_MACPL0_U16	VR62, VR5, VR4
	MVIL	VPERMREG, #0xb090301
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR4, VR5, VR4
	V_MUL_16	VR5, VR4, VR3
	V_MACPL0_U16	VR62, VR4, VR5
	V_PERMREG_16	VR5, VR63, VR62, VPERMREG
	V_ADD_16	VR4, VR4, VR5
	V_MUL_16	VR5, VR4, VR3
	V_MACPL0_U16	VR62, VR4, VR5
	V_PERMREG_16	VR5, VR63, VR62, VPERMREG
	V_ADD_16	VR4, VR4, VR5
	V_MUL_16	VR3, VR4, VR3
	V_MACPL0_U16	VR62, VR4, VR3
	V_PERMREG_16	VR3, VR63, VR62, VPERMREG
	V_ABSADDI_16	VR5, VR0, #0x0
	V_ADD_16	VR3, VR4, VR3
	V_MACPL0_U16	VR62, VR5, VR3
	V_PERMREG_16	VR3, VR63, VR62, VPERMREG
	V_MUL_16	VR4, VR3, VR2
	V_MVI_16	VR58, #0x0
	V_SUB_16	VR4, VR5, VR4
	V_SUBCS_16	VZERO, VR4, VR2
	MVI	VCONDSEL, #0xb
	V_MVI_16	VR5, #0x0
	V_MVCR_16	VR5, VR2
	V_SUB_16	VR4, VR4, VR5
	V_MV_16	VR5, VZERO
	V_ADDICR_16	VR5, VZERO, #0x1
	V_SUBCS_16	VZERO, VR4, VR2
	V_ADD_16	VR3, VR3, VR5
	V_MV_16	VR5, VZERO
	V_ADDICR_16	VR5, VZERO, #0x1
	V_XOR_16	VR59, VR1, VR0
	V_ADD_16	VR1, VR3, VR5
	V_MVCR_16	VR58, VR2
	V_SUB_16	VR2, VR4, VR58
	V_SUB_16	VR3, VZERO, VR1
	V_SUBICS_16	VZERO, VR59, #0x0
	MVI	VCONDSEL, #0x3
	SRI_U	SR0, SR0, #0x3
	V_MVCR_16	VR1, VR3
	V_SUB_16	VR3, VZERO, VR2
	V_SUBICS_16	VZERO, VR0, #0x0
	V_MVCR_16	VR2, VR3
	MV	VFIR0, SR0
	V_MV_16	(VFIR0), VR2
	V_MV_64	VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __divmod_16

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__mod_16                        // -- Begin function _mod_16
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__mod_16
	.org auto
//	.type	__mod_16, @function
__mod_16:                               // @_mod_16
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_16	VR1, VR1, #0x0
	V_SUB_16	VR2, VZERO, VR1
	V_CMBI_16	VR3, VR1, #0x0
	V_MVI_16	VR4, #0x1
	V_SL_16	VR4, VR4, VR3
	V_SL_16	VR3, VR2, VR3
	V_MACPL0_U16	VR62, VR4, VR3
	MVIL	VPERMREG, #0xb090301
	V_PERMREG_16	VR3, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR4, VR3
	V_MUL_16	VR4, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR4
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR3, VR4
	V_MUL_16	VR4, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR4
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR3, VR3, VR4
	V_MUL_16	VR2, VR3, VR2
	V_MACPL0_U16	VR62, VR3, VR2
	V_PERMREG_16	VR2, VR63, VR62, VPERMREG
	V_ABSADDI_16	VR4, VR0, #0x0
	V_ADD_16	VR2, VR3, VR2
	V_MACPL0_U16	VR62, VR4, VR2
	V_PERMREG_16	VR2, VR63, VR62, VPERMREG
	V_MUL_16	VR2, VR2, VR1
	V_SUB_16	VR2, VR4, VR2
	V_MVI_16	VR3, #0x0
	V_SUBCS_16	VZERO, VR2, VR1
	MVI	VCONDSEL, #0x0
	V_MV_64	VR4, VR1
	V_MVCR_16	VR4, VR3
	V_SUB_16	VR2, VR2, VR4
	V_SUBCS_16	VZERO, VR2, VR1
	V_MVCR_16	VR1, VR3
	V_SUB_16	VR1, VR2, VR1
	V_SUB_16	VR2, VZERO, VR1
	V_SUBICS_16	VZERO, VR0, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_16	VR1, VR2
	V_MV_64	VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __mod_16

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__div_16                        // -- Begin function _div_16
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__div_16
	.org auto
//	.type	__div_16, @function
__div_16:                               // @_div_16
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_16	VR2, VR1, #0x0
	V_SUB_16	VR3, VZERO, VR2
	V_CMBI_16	VR4, VR2, #0x0
	V_MVI_16	VR5, #0x1
	V_SL_16	VR5, VR5, VR4
	V_SL_16	VR4, VR3, VR4
	V_MACPL0_U16	VR62, VR5, VR4
	MVIL	VPERMREG, #0xb090301
	V_PERMREG_16	VR4, VR63, VR62, VPERMREG
	V_ADD_16	VR4, VR5, VR4
	V_MUL_16	VR5, VR4, VR3
	V_MACPL0_U16	VR62, VR4, VR5
	V_PERMREG_16	VR5, VR63, VR62, VPERMREG
	V_ADD_16	VR4, VR4, VR5
	V_MUL_16	VR5, VR4, VR3
	V_MACPL0_U16	VR62, VR4, VR5
	V_PERMREG_16	VR5, VR63, VR62, VPERMREG
	V_ADD_16	VR4, VR4, VR5
	V_MUL_16	VR3, VR4, VR3
	V_MACPL0_U16	VR62, VR4, VR3
	V_PERMREG_16	VR3, VR63, VR62, VPERMREG
	V_ABSADDI_16	VR5, VR0, #0x0
	V_ADD_16	VR3, VR4, VR3
	V_MACPL0_U16	VR62, VR5, VR3
	V_PERMREG_16	VR3, VR63, VR62, VPERMREG
	V_MUL_16	VR4, VR3, VR2
	V_SUB_16	VR4, VR5, VR4
	V_MVI_16	VR5, #0x0
	V_SUBCS_16	VZERO, VR4, VR2
	MVI	VCONDSEL, #0xb
	V_MVCR_16	VR5, VR2
	V_SUB_16	VR4, VR4, VR5
	V_MV_16	VR5, VZERO
	V_ADDICR_16	VR5, VZERO, #0x1
	V_SUBCS_16	VZERO, VR4, VR2
	V_ADD_16	VR2, VR3, VR5
	V_MV_16	VR3, VZERO
	V_ADDICR_16	VR3, VZERO, #0x1
	V_XOR_16	VR1, VR1, VR0
	V_ADD_16	VR0, VR2, VR3
	V_SUB_16	VR2, VZERO, VR0
	V_SUBICS_16	VZERO, VR1, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_16	VR0, VR2
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __div_16

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__divmod_8                      // -- Begin function _divmod_8
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__divmod_8
	.org auto
//	.type	__divmod_8, @function
__divmod_8:                             // @_divmod_8
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_8	VR2, VR1, #0x0
	V_SUB_8	VR3, VZERO, VR2
	V_CMBI_8	VR4, VR2, #0x0
	V_MVI_8	VR5, #0x1
	V_SL_8	VR5, VR5, VR4
	V_SL_8	VR4, VR3, VR4
	V_MACPL0_U8	VR62, VR5, VR4
	MVIL	VPERMREG, #-0x2468acf
	V_PERMREG_8	VR4, VR63, VR62, VPERMREG
	V_ADD_8	VR4, VR5, VR4
	V_MUL_8	VR5, VR4, VR3
	V_MACPL0_U8	VR62, VR4, VR5
	V_PERMREG_8	VR5, VR63, VR62, VPERMREG
	V_ADD_8	VR4, VR4, VR5
	V_MUL_8	VR3, VR4, VR3
	V_MACPL0_U8	VR62, VR4, VR3
	V_PERMREG_8	VR3, VR63, VR62, VPERMREG
	V_ABSADDI_8	VR5, VR0, #0x0
	V_ADD_8	VR3, VR4, VR3
	V_MACPL0_U8	VR62, VR5, VR3
	V_PERMREG_8	VR3, VR63, VR62, VPERMREG
	V_MUL_8	VR4, VR3, VR2
	V_MVI_8	VR58, #0x0
	V_SUB_8	VR4, VR5, VR4
	V_SUBCS_8	VZERO, VR4, VR2
	MVI	VCONDSEL, #0xb
	V_MVI_8	VR5, #0x0
	V_MVCR_8	VR5, VR2
	V_SUB_8	VR4, VR4, VR5
	V_MV_8	VR5, VZERO
	V_ADDICR_8	VR5, VZERO, #0x1
	V_SUBCS_8	VZERO, VR4, VR2
	V_ADD_8	VR3, VR3, VR5
	V_MV_8	VR5, VZERO
	V_ADDICR_8	VR5, VZERO, #0x1
	V_XOR_8	VR59, VR1, VR0
	V_ADD_8	VR1, VR3, VR5
	V_MVCR_8	VR58, VR2
	V_SUB_8	VR2, VR4, VR58
	V_SUB_8	VR3, VZERO, VR1
	V_SUBICS_8	VZERO, VR59, #0x0
	MVI	VCONDSEL, #0x3
	SRI_U	SR0, SR0, #0x3
	V_MVCR_8	VR1, VR3
	V_SUB_8	VR3, VZERO, VR2
	V_SUBICS_8	VZERO, VR0, #0x0
	V_MVCR_8	VR2, VR3
	MV	VFIR0, SR0
	V_MV_8	(VFIR0), VR2
	V_MV_64	VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __divmod_8

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__mod_8                         // -- Begin function _mod_8
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__mod_8
	.org auto
//	.type	__mod_8, @function
__mod_8:                                // @_mod_8
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_8	VR1, VR1, #0x0
	V_SUB_8	VR2, VZERO, VR1
	V_CMBI_8	VR3, VR1, #0x0
	V_MVI_8	VR4, #0x1
	V_SL_8	VR4, VR4, VR3
	V_SL_8	VR3, VR2, VR3
	V_MACPL0_U8	VR62, VR4, VR3
	MVIL	VPERMREG, #-0x2468acf
	V_PERMREG_8	VR3, VR63, VR62, VPERMREG
	V_ADD_8	VR3, VR4, VR3
	V_MUL_8	VR4, VR3, VR2
	V_MACPL0_U8	VR62, VR3, VR4
	V_PERMREG_8	VR4, VR63, VR62, VPERMREG
	V_ADD_8	VR3, VR3, VR4
	V_MUL_8	VR2, VR3, VR2
	V_MACPL0_U8	VR62, VR3, VR2
	V_PERMREG_8	VR2, VR63, VR62, VPERMREG
	V_ABSADDI_8	VR4, VR0, #0x0
	V_ADD_8	VR2, VR3, VR2
	V_MACPL0_U8	VR62, VR4, VR2
	V_PERMREG_8	VR2, VR63, VR62, VPERMREG
	V_MUL_8	VR2, VR2, VR1
	V_SUB_8	VR2, VR4, VR2
	V_MVI_8	VR3, #0x0
	V_SUBCS_8	VZERO, VR2, VR1
	MVI	VCONDSEL, #0x0
	V_MV_64	VR4, VR1
	V_MVCR_8	VR4, VR3
	V_SUB_8	VR2, VR2, VR4
	V_SUBCS_8	VZERO, VR2, VR1
	V_MVCR_8	VR1, VR3
	V_SUB_8	VR1, VR2, VR1
	V_SUB_8	VR2, VZERO, VR1
	V_SUBICS_8	VZERO, VR0, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_8	VR1, VR2
	V_MV_64	VR0, VR1
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __mod_8

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.export	__div_8                         // -- Begin function _div_8
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
	.function	__div_8
	.org auto
//	.type	__div_8, @function
__div_8:                                // @_div_8
                                        // emit label via emitLabel
// %bb.0:                               // Function body start
	ADDFIRI	SFIR3, SFIR3, #-0x2
	V_ABSADDI_8	VR2, VR1, #0x0
	V_SUB_8	VR3, VZERO, VR2
	V_CMBI_8	VR4, VR2, #0x0
	V_MVI_8	VR5, #0x1
	V_SL_8	VR5, VR5, VR4
	V_SL_8	VR4, VR3, VR4
	V_MACPL0_U8	VR62, VR5, VR4
	MVIL	VPERMREG, #-0x2468acf
	V_PERMREG_8	VR4, VR63, VR62, VPERMREG
	V_ADD_8	VR4, VR5, VR4
	V_MUL_8	VR5, VR4, VR3
	V_MACPL0_U8	VR62, VR4, VR5
	V_PERMREG_8	VR5, VR63, VR62, VPERMREG
	V_ADD_8	VR4, VR4, VR5
	V_MUL_8	VR3, VR4, VR3
	V_MACPL0_U8	VR62, VR4, VR3
	V_PERMREG_8	VR3, VR63, VR62, VPERMREG
	V_ABSADDI_8	VR5, VR0, #0x0
	V_ADD_8	VR3, VR4, VR3
	V_MACPL0_U8	VR62, VR5, VR3
	V_PERMREG_8	VR3, VR63, VR62, VPERMREG
	V_MUL_8	VR4, VR3, VR2
	V_SUB_8	VR4, VR5, VR4
	V_MVI_8	VR5, #0x0
	V_SUBCS_8	VZERO, VR4, VR2
	MVI	VCONDSEL, #0xb
	V_MVCR_8	VR5, VR2
	V_SUB_8	VR4, VR4, VR5
	V_MV_8	VR5, VZERO
	V_ADDICR_8	VR5, VZERO, #0x1
	V_SUBCS_8	VZERO, VR4, VR2
	V_ADD_8	VR2, VR3, VR5
	V_MV_8	VR3, VZERO
	V_ADDICR_8	VR3, VZERO, #0x1
	V_XOR_8	VR1, VR1, VR0
	V_ADD_8	VR0, VR2, VR3
	V_SUB_8	VR2, VZERO, VR0
	V_SUBICS_8	VZERO, VR1, #0x0
	MVI	VCONDSEL, #0x3
	V_MVCR_8	VR0, VR2
	ADDFIRI	SFIR3, SFIR3, #0x2
	JLA	ZERO, R31
	//	.endfunction	// __div_8

	.endfunction
	.endsection


	.csection	VMPCC_CODE_AND_DATA_18022025_090130_981754042_2760102
                                        // Function body end
                                        // -- End function
	.endsection
// THAT'S ALL FOLKS...
	.end

/// @endcond

