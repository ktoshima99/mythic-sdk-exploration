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
 * FILENAME:      msg.asm
 *
 * DESCRIPTION:   message data structure definition
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/


/*	[       rsvd |       rsvd |      rsvd  |flags| tcnt ]
	[       numF |       numW |     mpResH |       numH ]
	[   inpBaseL |   inpBaseR |  softpFirst|  softpRest ]
	[           wgts_ext_addr |            |   wgts_len ]
	[          cdata_ext_addr |            |  cdata_len ]

	[         build_timestamp |      debug |   vmp_code ]

	[                  descr_load_wgts                  ]
	[                  descr_load_cdata                 ]
	[                  descr_pad_0                      ]
	[                  descr_pad_1                      ]
	[                  descr_pad_2                      ]
	[                  descr_pad_3                      ]
	[                  descr_load_inp_first             ]
	[                  descr_load_inp_rest              ]
	[                  descr_preload_wgts               ]
	[                  descr_store_out                  ]
	[         ext_addr_start  |            ext_addr_inc ]
	[  dec_cnt[1]|int_addr[1] | int_addr[2]|int_addr[3] ]  // NOTE: int_addr[0] = dec_cnt[1] (see below)
	[ int_addr[4]|int_addr[5] |   outBaseL |   outBaseR ]

	[                   descr_load_aux                  ]
	[          load_aux_start |            load_aux_inc ]
	[  dec_cnt[1]| int_addr[1]| int_addr[2]| int_addr[3]]	// NOTE: int_addr[0] = dec_cnt[1] (see below)
	[ int_addr[4]| int_addr[5]|   auxBaseL |   auxBaseR ]
	[     ovl_extstr          | ovl_intsrc |    ovl_len ]

	NOTE: dec_cnt[1] is a special field for convolutional layers with numH=1:
	In this mode, the input row is split horizontally into two parts each of width numWPad/2
	These parts are computed in parallel. The output is transferred using two separate DMA channels
	(and descriptors).

	If numW < numWPad then the "count" field of the second DMA descriptor has to be decremented
	accordingly. This is achieved by subtracting dec_cnt[1] from the count field of the second output
	DMA descriptor.

	<--------- numW ------>
	  numWPad/2    numWPad/2
	<----------> <---------->
	|  part 0   |   part 1  |


	int_addr[0] is not used otherwise, the very first int_addr is taken directly from
	descr_store_out.int_addr.

	*/

//NOTE: .equ definitions moved to common file main_defs.asm


// layer_message structure
.dsection layer_msg, dmem
.org auto
start:

	// 64       56       48       40       32       24       16        8        0
	//  |      msgbuf_base|      cdata_base |flags   |num_tpls|       num_parts |
	// num_tpls = num_msg_templates

	.alloc data
	.alloc dummy // FIXME: watch data alignment when extending this structure!
	.equ SIZE     = 0x1
.endsection



// message buffer compression: partition_message structure

.dsection PART_DIFF, dmem

	.equ AUX_LOAD_EXT_BASE                               = 0
	.equ OUT_STORE_EXT_BASE                              = 1
	.equ LOAD_INP_FIRST_EXT_BASE                         = 2
	.equ LOAD_INP_REST_EXT_BASE                          = 3
	.equ WGTS_EXT_ADDR                                   = 4
	.equ CDATA_EXT_ADDR                                  = 5
	.equ WGTS_LEN__CDATA_LEN__IDX__TEMPLATE_ID           = 6

	.equ SIZE = 7

.org auto
	.alloc MSG[SIZE]
.endsection
