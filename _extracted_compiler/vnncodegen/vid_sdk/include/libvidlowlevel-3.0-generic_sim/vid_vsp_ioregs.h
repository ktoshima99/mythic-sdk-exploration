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
 * FILENAME:             vid_vsp_ioregs.h
 *
 * DESCRIPTION:          v-SP3 IO register map
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef _VID_VSP_IOREGS_H_
#define _VID_VSP_IOREGS_H_

/**
 * @brief videantis LowLevel Library vid_vsp_ioregs.h include file
 *
 * @details
 * Address definitions of v-SP internal IO registers
 *
 * @file vid_vsp_ioregs.h
 */

#define VSP_LRM_BASE	0xC00000
#define VSP_IRQVEC_BASE 0xC000E0
#define VSP_BIT_OFFSET	32

#define VSP_ADDR_GPDATA0	((VSP_LRM_BASE + 0x140) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA1	((VSP_LRM_BASE + 0x141) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA2	((VSP_LRM_BASE + 0x142) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA3	((VSP_LRM_BASE + 0x143) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA4	((VSP_LRM_BASE + 0x144) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA5	((VSP_LRM_BASE + 0x145) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA6	((VSP_LRM_BASE + 0x146) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA7	((VSP_LRM_BASE + 0x147) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA8	((VSP_LRM_BASE + 0x148) * VSP_BIT_OFFSET)
#define VSP_ADDR_GPDATA9	((VSP_LRM_BASE + 0x149) * VSP_BIT_OFFSET)
#define VSP_ADDR_DCI_IRQIC	((VSP_LRM_BASE + 0x14e) * VSP_BIT_OFFSET)
#define VSP_ADDR_DCI_IRQOUT	((VSP_LRM_BASE + 0x14f) * VSP_BIT_OFFSET)
#define VSP_ADDR_DCI_IRQ0	((VSP_LRM_BASE + 0x150) * VSP_BIT_OFFSET)
#define VSP_ADDR_DCI_IRQ1	((VSP_LRM_BASE + 0x151) * VSP_BIT_OFFSET)
#define VSP_ADDR_DCI_IRQIN	((VSP_LRM_BASE + 0x152) * VSP_BIT_OFFSET)
#define VSP_ADDR_DCI_IRQINCTRL	((VSP_LRM_BASE + 0x153) * VSP_BIT_OFFSET)
#define VSP_ADDR_DCI_IRQEN	((VSP_LRM_BASE + 0x154) * VSP_BIT_OFFSET)
#define VSP_ADDR_DCI_IRQSTATUS	((VSP_LRM_BASE + 0x155) * VSP_BIT_OFFSET)
#define VSP_ADDR_SP_VERSION	((VSP_LRM_BASE + 0x157) * VSP_BIT_OFFSET)

#define VSP_ADDR_IRQVEC_IRQIN  ((VSP_IRQVEC_BASE + 24) * VSP_BIT_OFFSET)
#define VSP_ADDR_IRQCTRL       ((VSP_LRM_BASE + 50) * VSP_BIT_OFFSET)

#define VSP_DCI_IRQINCTRL_IRQINSEL_POSEDGE 0x00
#define VSP_DCI_IRQINCTRL_IRQINSEL_NEGEDGE 0x01
#define VSP_DCI_IRQINCTRL_IRQINSEL_HIGH    0x02
#define VSP_DCI_IRQINCTRL_IRQINSEL_LOW     0x03

#define VSP_DCI_IRQSTATUS_GPD0FLAG    0
#define VSP_DCI_IRQSTATUS_GPD1FLAG    1
#define VSP_DCI_IRQSTATUS_GPD2FLAG    2
#define VSP_DCI_IRQSTATUS_GPD3FLAG    3
#define VSP_DCI_IRQSTATUS_GPD4FLAG    4
#define VSP_DCI_IRQSTATUS_GPD5FLAG    5
#define VSP_DCI_IRQSTATUS_GPD6FLAG    6
#define VSP_DCI_IRQSTATUS_GPD7FLAG    7
#define VSP_DCI_IRQSTATUS_GPD8FLAG    8
#define VSP_DCI_IRQSTATUS_GPD9FLAG    9
#define VSP_DCI_IRQSTATUS_IRQOUTFLAG 15
#define VSP_DCI_IRQSTATUS_IRQ0FLAG   16
#define VSP_DCI_IRQSTATUS_IRQ1FLAG   17
#define VSP_DCI_IRQSTATUS_IRQINFLAG  18

#endif
