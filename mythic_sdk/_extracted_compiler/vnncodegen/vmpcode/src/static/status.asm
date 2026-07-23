/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2026 videantis GmbH
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
 * FILENAME:      status.asm
 *
 * DESCRIPTION:   Very simple v-MP->Host Error/Status signalling using GPDATA7
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

.equ VMP_READY   = 0
.equ VMP_BUSY    = 1
.equ VMP_ERROR	 = 2

.equ VMP_ERR_VERSION = 0x00000001

.macro raise_error(n)
	MVIL	SR0, #n
	JLR	ZERO, error.raise
	NOP
	NOP
.endmacro

.csection error
.scheduling on
raise:
	MVI	GPDATA7, #VMP_ERROR
	MV	GPDATA1, SR0
	SLEEP	0xffff
	JLR	ZERO, error.raise
.endsection



