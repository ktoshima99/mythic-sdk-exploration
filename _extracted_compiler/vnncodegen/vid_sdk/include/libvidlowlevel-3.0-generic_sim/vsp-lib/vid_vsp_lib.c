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
written authorization of videantis GmbH or an individual license agreementp
with videantis GmbH is strictly forbidden.

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
*
* FILENAME: vid_vsp_lib.c
*
* DESCRIPTION: support for file IO and other semihosting features
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#include "vid_vsp_lib.h"

#ifndef EXCLUDE_MEMSET
void *memset(void *mem,int c,size_t size) {
  size_t pos=0;
  char *string=(char*)mem;
  for(pos=0;pos<size;pos++) string[pos]=(char)c;
  return mem;
}
#endif

#ifndef EXCLUDE_STRLEN
size_t strlen(const char *p) {
  size_t l=0;
  while(*p++) l++;
  return l;
}
#endif

#ifndef EXCLUDE_STRCPY
#ifdef __GNUC__
char *strcpy(char *dst, const char *src) {
#else
char *strcpy(__bitaddress char *dst, __bitaddress const char *src) {
#endif
  char *r=dst;
  do {
    *dst++=*src;
  } while(*src++);
  return r;
}
#endif

#ifndef EXCLUDE_ATOI
int atoi(const char *str) {
  // note: no overflow check
  int sign=1,val=0;
  const int base=10;
  if(*str=='-') {
    str++; sign=-1;
  }
  while(*str>='0' && *str<='9') {
    val=val*base+*str-'0';
    str++;
  }
  return val*sign;
}
#endif
