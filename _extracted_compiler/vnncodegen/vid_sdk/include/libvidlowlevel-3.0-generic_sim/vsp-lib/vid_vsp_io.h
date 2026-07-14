/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2023 videantis GmbH
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
 * FILENAME: vid_vsp_io.h
 *
 * DESCRIPTION: support for file IO and other semihosting features
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#ifndef _VID_VSP_IO_H
#define _VID_VSP_IO_H

// addresses selected for chip-it compatibility
#define SYSCALL_END_SIM     0x0004  // write here to stop simulation (value 0 for OK, other value for error)

#define SYSCALL_DBGOUT_HI   0x0010  // 64-bit hex debug output; even words: store high word
#define SYSCALL_DBGOUT_LO   0x0014  // 64-bit hex debug output; odd words: output 64-bit word
#define SYSCALL_DBGOUT_END  0x0800  // end address (excl.) used by debug output function

#define SYSCALL_IMGREADY_HI 0x2000  // write data to output file specified by parameter outfile: first write size (bytes) here,
#define SYSCALL_IMGREADY_LO 0x2004  //  then write address here to start output; reading returns always zero
// if profiling is enabled, writing to IMGREADY_LO also triggers profile output

#define SYSCALL_DUMP_HI     0x2100  // dump data to standard output: first write size (bytes) here,
#define SYSCALL_DUMP_LO     0x2104  //  then write address here to start output; reading returns always zero

// simulator-specific functions
#define SYSCALL_TRACE       0x7104  // controls trace output (write only; 1: on, 0: off)
#define SYSCALL_OUTPUT_PROF 0x710c  // write here to output profile counters and reset them (if profiling is enabled)

#define SYSCALL_CYCLE_HI    0x7110  // cycle timer, high word (read only)
#define SYSCALL_CYCLE       0x7114  // cycle timer, low word (read only)
#define SYSCALL_CYCLE_HI_L  0x711c  // cycle timer, upper 32 bits (read only, latched when reading low word!)

#define SYSCALL_PRINT_CHAR  0x7124  // write character to be printed
#define SYSCALL_PRINT_STR   0x712c  // write c-string pointer to be printed

#define SYSCALL_FILE_OPEN   0x7144  // write 1 to open file (after setting parameters); read file handle from here
#define SYSCALL_FO_NAME     0x714c  // file name pointer for SYSCALL_FILE_OPEN
#define SYSCALL_FO_MODE     0x7154  // file mode string pointer for SYSCALL_FILE_OPEN

#define SYSCALL_FILE_CLOSE  0x715c  // write handle here to close file

#define SYSCALL_FILE_SEEK   0x7164  // write handle here to execute fseek(), then read result
#define SYSCALL_FS_OFFSET   0x716c  // fseek offset
#define SYSCALL_FS_ORIGIN   0x7174  // fseek origin (SEEK_SET=0, SEEK_CUR=1, SEEK_END=2)

#define SYSCALL_FILE_TELL   0x717c  // write handle here to execute ftell(), read result from here

#define SYSCALL_FILE_READ   0x7184  // write handle here to start fread operation, read number of elements transferred
#define SYSCALL_FR_PTR      0x718c  // buffer pointer for SYSCALL_FILE_READ
#define SYSCALL_FR_SIZE     0x7194  // element size for SYSCALL_FILE_READ
#define SYSCALL_FR_COUNT    0x719c  // element count for SYSCALL_FILE_READ

#define SYSCALL_FILE_WRITE  0x71a4  // write handle here to start fwrite operation, read number of elements transferred
#define SYSCALL_FW_PTR      0x71ac  // buffer pointer for SYSCALL_FILE_WRITE
#define SYSCALL_FW_SIZE     0x71b4  // element size for SYSCALL_FILE_WRITE
#define SYSCALL_FW_COUNT    0x71bc  // element count for SYSCALL_FILE_WRITE

#define SYSCALL_PRINTF_VAR_HI   0x71c0  // first write variables to these addresses (first HI, then LO),
#define SYSCALL_PRINTF_VAR_LO   0x71c4
#define SYSCALL_PRINTF_STR      0x71cc  // then write host memory string address here
#define SYSCALL_PRINTF_STR_DMEM 0x71dc  // or write DMEM string address here

#define SYSCALL_ARGC        0x7800  // argc (read only)
#define SYSCALL_ARGV        0x7804  // argv[] (read only, argv[0] always points to an empty string)
// this range is reserved for argv[] and the argument c-strings (read only)

#define SEMIHOST_ADDR_MASK  0x7fff  // address mask (i.e. size-1) of host address window occupied by semihosting functions

// direct printf feature without DMA
#define PRINTF_SEMIHOST_BASE_W 0x7fc000 // internal word address, to be used without DMA


// support for file IO and other semihosting features
#include "vid_vsp_lib.h"

#ifndef SEMIHOSTING_BUF
 #define SEMIHOSTING_BUF    0x0a000000  // buffer base in HMEM (for string/data transfer)
#endif
#define SEMIHOSTING_BUF_LEN 0x8000      // buffer length

#ifndef SEMIHOSTING_ADDR
 #define SEMIHOSTING_ADDR   0x1fff0000  // address of semihosting interface
#endif

void dbgout(unsigned idx, unsigned hi, unsigned lo);

#define DBGOUT(VAL_HI,VAL_LO)      __asm__ volatile ("mv_dd 0x7fc004w,@%0                       \n\tmv_dd 0x7fc005w,@%1\n\t"               :: "Rx" (VAL_HI), "Rx" (VAL_LO))
#define DBGOUTI(IDX,VAL_HI,VAL_LO) __asm__ volatile ("mv_dd {0x7fc002w+{"#IDX"}*64},@%0         \n\tmv_dd {0x7fc003w+{"#IDX"}*64},@%1\n\t" :: "Rx" (VAL_HI), "Rx" (VAL_LO))
#define DBGOUT_BSR(VAL_LO)         __asm__ volatile ("mv_dd 0x7fc004w,@aBitstreamR              \n\tmv_dd 0x7fc005w,@%0\n\t"               :: "Rx" (VAL_LO))
#define DBGOUTI_BSR(IDX,VAL_LO)    __asm__ volatile ("mv_dd {0x7fc002w+{"#IDX"}*64},@aBitstreamR\n\tmv_dd {0x7fc003w+{"#IDX"}*64},@%0\n\t" :: "Rx" (VAL_LO))

typedef unsigned FILE; // dummy FILE type (FILE* could also be void*)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
FILE *fopen(const char *filename, const char *mode);
int fclose(FILE *file);
unsigned fread(void *buffer, unsigned size, unsigned count, FILE *file);
unsigned fwrite(const void *buffer, unsigned size, unsigned count, FILE *file);
int fseek(FILE *file, long offset, int origin);
long ftell(FILE *file);

unsigned fread_hmem(unsigned addr, unsigned size, unsigned count, FILE *file);
unsigned fwrite_hmem(unsigned addr, unsigned size, unsigned count, FILE *file);

int puts(const char *str);
#ifdef __GNUC__
int printf(const char *format, ...);
#else
int printf(__bitaddress const char *format, ...);
#endif
int putchar(int c);

#endif
