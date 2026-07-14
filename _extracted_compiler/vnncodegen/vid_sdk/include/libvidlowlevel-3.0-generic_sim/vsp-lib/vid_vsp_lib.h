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
 written authorization of videantis GmbH or an individual license agreementp
 with videantis GmbH is strictly forbidden.

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
*
* FILENAME: vid_vsp_lib.h
*
* DESCRIPTION: support for file IO and other semihosting features
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef _VID_VSP_LIB_H
#define _VID_VSP_LIB_H

#define NULL 0

typedef unsigned size_t;

int vid_vsp_mem_init(unsigned int addr, unsigned int size);
int vid_vsp_ocmem_init(unsigned int addr, unsigned int size);
int vid_mbox_init_vmp(unsigned core_id, unsigned local_mailbox_base);

// DMA functions using DMA channel 15
void dma_write(unsigned addr,unsigned data);
void dma_write64(unsigned addr,unsigned data_hi,unsigned data_lo);
void dma_wait();
unsigned dma_read(unsigned addr);
void dma_write_block(unsigned ext_addr,const char *int_ptr,unsigned len);
void dma_read_block(unsigned ext_addr,char *int_ptr,unsigned len);
void dma_read_block_nowait(unsigned ext_addr,char *int_ptr,unsigned len);
void dma_memset_ext(unsigned ext_addr,char c,unsigned len);
void dma_memset_int(char* int_ptr,char c,unsigned len);

// DMA functions using alternate DMA channel 13
void dma_write_chB(unsigned addr,unsigned data);
void dma_write64_chB(unsigned addr,unsigned data_hi,unsigned data_lo);
void dma_wait_chB();
unsigned dma_read_chB(unsigned addr);
void dma_write_block_chB(unsigned ext_addr,const char *int_ptr,unsigned len);
void dma_read_block_chB(unsigned ext_addr,char *int_ptr,unsigned len);
void dma_read_block_nowait_chB(unsigned ext_addr,char *int_ptr,unsigned len);
void dma_memset_ext_chB(unsigned ext_addr,char c,unsigned len);
void dma_memset_int_chB(char* int_ptr,char c,unsigned len);

void dma_wait_all_ch();    // wait until all DMA transfers started by C functions have finished

void *memset(void*,int,size_t);
void *memcpy(void*,const void*,size_t);
void *memcpyw(void*,const void*,size_t);  // optimized version for aligned words only
size_t strlen(const char*);
#ifdef __GNUC__
char *strcpy(char*, const char*);
#else
char *strcpy(__bitaddress char*, __bitaddress const char*);
#endif
int atoi(const char *str);

void exit(int);


// inline assembly macros
#define vid_nop __asm__ volatile("nop");

#define vid_clz(Xzeros,Xlookahead) __asm__ volatile ("cmb_www  %0, %1, zero" : "=Rx" (Xzeros) : "Rx" (Xlookahead))
#define vid_abs(Xdst, Xsrc) __asm__ volatile("abs  %0, %1" : "=Rx" (Xdst) : "Rx" (Xsrc))

#define vid_minu(Xdst, Xsrc1, Xsrc2) __asm__ volatile("minu  %0, %1, %2" : "=Rx" (Xdst) : "Rx" (Xsrc1), "Rx" (Xsrc2))
#define vid_maxu(Xdst, Xsrc1, Xsrc2) __asm__ volatile("maxu  %0, %1, %2" : "=Rx" (Xdst) : "Rx" (Xsrc1), "Rx" (Xsrc2))
#define vid_mins(Xdst, Xsrc1, Xsrc2) __asm__ volatile("mins  %0, %1, %2" : "=Rx" (Xdst) : "Rx" (Xsrc1), "Rx" (Xsrc2))
#define vid_maxs(Xdst, Xsrc1, Xsrc2) __asm__ volatile("maxs  %0, %1, %2" : "=Rx" (Xdst) : "Rx" (Xsrc1), "Rx" (Xsrc2))

#define vid_minui(Xdst, Xsrc, Ximm) __asm__ volatile("minu  %0, %1, #" #Ximm : "=Rx" (Xdst) : "Rx" (Xsrc))
#define vid_maxui(Xdst, Xsrc, Ximm) __asm__ volatile("maxu  %0, %1, #" #Ximm : "=Rx" (Xdst) : "Rx" (Xsrc))
#define vid_minsi(Xdst, Xsrc, Ximm) __asm__ volatile("mins  %0, %1, #" #Ximm : "=Rx" (Xdst) : "Rx" (Xsrc))
#define vid_maxsi(Xdst, Xsrc, Ximm) __asm__ volatile("maxs  %0, %1, #" #Ximm : "=Rx" (Xdst) : "Rx" (Xsrc))

#define vid_wait_irq_dci0 __asm__ volatile ("wait 0x10200");

#endif
