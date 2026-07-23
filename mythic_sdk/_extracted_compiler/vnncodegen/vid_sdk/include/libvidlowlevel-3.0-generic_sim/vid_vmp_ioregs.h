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
 * FILENAME:    vid_vmp_ioregs.h
 *
 * DESCRIPTION: Address definitions of v-MP internal IO registers
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef _VID_VMP_IOREGS_H_
#define _VID_VMP_IOREGS_H_

/**
 * @brief videantis LowLevel Library vid_vmp_ioregs.h include file
 *
 * @details
 * Address definitions of v-MP internal IO registers
 *
 * @file vid_vmp_ioregs.h
 */

#define VMP_ADDR_BOOT_ADDR_L 0x3BEu
#define VMP_ADDR_BOOT_ADDR_H 0x3BFu

#define VMP_ADDR_TIMER1    0x3D0u
#define VMP_ADDR_TCTRL1    0x3D1u
#define VMP_ADDR_TIMER2    0x3D2u
#define VMP_ADDR_TCTRL2    0x3D3u

#define VMP_ADDR_IRQIC     0x3D6u
#define VMP_ADDR_IRQIN     0x3D8u
#define VMP_ADDR_IRQOUT    0x3D9u

#define VMP_ADDR_GPDATA0   0x3DAu
#define VMP_ADDR_GPDATA1   0x3DBu
#define VMP_ADDR_GPDATA2   0x3DCu
#define VMP_ADDR_GPDATA3   0x3DDu
#define VMP_ADDR_GPDATA4   0x3DEu
#define VMP_ADDR_GPDATA5   0x3DFu
#define VMP_ADDR_GPDATA6   0x3E0u
#define VMP_ADDR_GPDATA7   0x3E1u
#define VMP_ADDR_GPD_FLAGS 0x3E2u
#define VMP_ADDR_GPDFLAGS  0x3E2u


#define VMP_BADDR_BOOT_ADDR_L (VMP_ADDR_BOOT_ADDR_L << 3u)
#define VMP_BADDR_BOOT_ADDR_H (VMP_ADDR_BOOT_ADDR_H << 3u)

#define VMP_BADDR_TIMER1    (VMP_ADDR_TIMER1 << 3u)
#define VMP_BADDR_TCTRL1    (VMP_ADDR_TCTRL1 << 3u)
#define VMP_BADDR_TIMER2    (VMP_ADDR_TIMER2 << 3u)
#define VMP_BADDR_TCTRL2    (VMP_ADDR_TCTRL2 << 3u)

#define VMP_BADDR_IRQIC     (VMP_ADDR_IRQIC << 3u)
#define VMP_BADDR_IRQIN     (VMP_ADDR_IRQIN << 3u)
#define VMP_BADDR_IRQOUT    (VMP_ADDR_IRQOUT << 3u)

#define VMP_BADDR_GPDATA0   (VMP_ADDR_GPDATA0 << 3u)
#define VMP_BADDR_GPDATA1   (VMP_ADDR_GPDATA1 << 3u)
#define VMP_BADDR_GPDATA2   (VMP_ADDR_GPDATA2 << 3u)
#define VMP_BADDR_GPDATA3   (VMP_ADDR_GPDATA3 << 3u)
#define VMP_BADDR_GPDATA4   (VMP_ADDR_GPDATA4 << 3u)
#define VMP_BADDR_GPDATA5   (VMP_ADDR_GPDATA5 << 3u)
#define VMP_BADDR_GPDATA6   (VMP_ADDR_GPDATA6 << 3u)
#define VMP_BADDR_GPDATA7   (VMP_ADDR_GPDATA7 << 3u)
#define VMP_BADDR_GPD_FLAGS (VMP_ADDR_GPD_FLAGS << 3u)
#define VMP_BADDR_GPDFLAGS  (VMP_ADDR_GPDFLAGS << 3u)


#define VMP_WAIT_MODE_GPDATA  (1u << 21u)
#define VMP_WAIT_MODE_DMA     (0u << 21u)
#define VMP_WAIT_GPDATA0      (1u <<  0u)
#define VMP_WAIT_GPDATA1      (1u <<  1u)
#define VMP_WAIT_GPDATA2      (1u <<  2u)
#define VMP_WAIT_GPDATA3      (1u <<  3u)
#define VMP_WAIT_GPDATA4      (1u <<  4u)
#define VMP_WAIT_GPDATA5      (1u <<  5u)
#define VMP_WAIT_GPDATA6      (1u <<  6u)
#define VMP_WAIT_GPDATA7      (1u <<  7u)
#define VMP_WAIT_IRQOUT_SET   (1u << 16u)
#define VMP_WAIT_IRQOUT_CLEAR (1u << 17u)
#define VMP_WAIT_IRQIN_SET    (1u << 18u)

#define VMP_CTRL_ADDR_MASK 0xfffffff8u

#endif //_VID_VMP_IOREGS_H_
