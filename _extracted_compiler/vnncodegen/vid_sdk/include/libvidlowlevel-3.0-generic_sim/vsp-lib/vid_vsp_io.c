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
 * FILENAME: vid_vsp_io.c
 *
 * DESCRIPTION: support for file IO and other semihosting features
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#include "vid_vsp_io.h"
#include <stdarg.h>

char print_str_global[2];

FILE *fopen(const char *filename, const char *mode)
{
    unsigned name_len = strlen(filename) + 1, mode_len = strlen(mode) + 1;
    dma_write_block(SEMIHOSTING_BUF, filename, name_len);
    dma_write_block(SEMIHOSTING_BUF + name_len, mode, mode_len);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FO_NAME, SEMIHOSTING_BUF);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FO_MODE, SEMIHOSTING_BUF + name_len);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FILE_OPEN, 1);
    return (FILE *)dma_read(SEMIHOSTING_ADDR + SYSCALL_FILE_OPEN);
}

int fclose(FILE *file)
{
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FILE_CLOSE, (unsigned)file);
    return 0;
}

unsigned fread(void *buffer, unsigned size, unsigned count, FILE *file)
{
    unsigned len;
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FR_PTR, SEMIHOSTING_BUF);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FR_SIZE, size);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FR_COUNT, count);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FILE_READ, (unsigned)file);
    len = dma_read(SEMIHOSTING_ADDR + SYSCALL_FILE_READ);
    dma_read_block(SEMIHOSTING_BUF, buffer, len * size);
    return len;
}

unsigned fread_hmem(unsigned addr, unsigned size, unsigned count, FILE *file)
{
    unsigned len;
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FR_PTR, addr);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FR_SIZE, size);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FR_COUNT, count);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FILE_READ, (unsigned)file);
    len = dma_read(SEMIHOSTING_ADDR + SYSCALL_FILE_READ);
    return len;
}

unsigned fwrite(const void *buffer, unsigned size, unsigned count, FILE *file)
{
    dma_write_block(SEMIHOSTING_BUF, buffer, size * count);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FW_PTR, SEMIHOSTING_BUF);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FW_SIZE, size);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FW_COUNT, count);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FILE_WRITE, (unsigned)file);
    return dma_read(SEMIHOSTING_ADDR + SYSCALL_FILE_WRITE);
}

unsigned fwrite_hmem(unsigned addr, unsigned size, unsigned count, FILE *file)
{
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FW_PTR, addr);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FW_SIZE, size);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FW_COUNT, count);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FILE_WRITE, (unsigned)file);
    return dma_read(SEMIHOSTING_ADDR + SYSCALL_FILE_WRITE);
}

int fseek(FILE *file, long offset, int origin)
{
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FS_OFFSET, (unsigned)offset);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FS_ORIGIN, (unsigned)origin);
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FILE_SEEK, (unsigned)file);
    return (int)dma_read(SEMIHOSTING_ADDR + SYSCALL_FILE_SEEK);
}

long ftell(FILE *file)
{
    dma_write(SEMIHOSTING_ADDR + SYSCALL_FILE_TELL, (unsigned)file);
    return (long)dma_read(SEMIHOSTING_ADDR + SYSCALL_FILE_TELL);
}

__attribute__((always_inline)) inline void printf_semihost_write(unsigned offset, unsigned value)
{
    volatile unsigned * const ptr = (volatile unsigned * const) (PRINTF_SEMIHOST_BASE_W * 32u + (offset & SEMIHOST_ADDR_MASK) * 8u);
    *ptr = value;
}

int putchar(int c)
{
    print_str_global[0] = c;
    print_str_global[1] = '\0';
    printf_semihost_write(SEMIHOSTING_ADDR + SYSCALL_PRINTF_STR_DMEM, (unsigned)print_str_global); 

    return c;
}

int puts(const char *str)
{
    while (*str)
        putchar(*str++);
    putchar('\n');
    return 0;
}

void dbgout(unsigned idx, unsigned hi, unsigned lo)
{
    vid_maxui(idx, idx, 1);
    vid_minui(idx, idx, 0x7f);
    dma_write64(SEMIHOSTING_ADDR + SYSCALL_DBGOUT_HI + ((idx - 1) << 3), hi, lo);
}



#ifdef __GNUC__
int printf(const char *format, ...)
{
#else
int printf(__bitaddress const char *format, ...)
{
#endif
    int i, count = 0;
    const char *format_begin = format;
    va_list arglist;
    va_start(arglist, format);

    // count number of printf arguments
    while (*format) {
        if (*format == '%' && *(format + 1) == '%') {
            format += 2;
        } else if (*format == '%') { // format string?
            format++;
            count++;
        } else {
            format++;
        }
    }

    for (i = 0; i < count; i++) {
        unsigned u = va_arg(arglist, unsigned);
        printf_semihost_write(SEMIHOSTING_ADDR + SYSCALL_PRINTF_VAR_LO, u);
    }

#ifdef __GNUC__
    printf_semihost_write(SEMIHOSTING_ADDR + SYSCALL_PRINTF_STR_DMEM, (unsigned)(format_begin));
#else
    printf_semihost_write(SEMIHOSTING_ADDR + SYSCALL_PRINTF_STR_DMEM, (unsigned)__builtin_vsp_convert_byteaddresstobitaddress(format_begin));
#endif

    va_end(arglist);
    return count;
}

int *__errno()
{
    return 0;
}
