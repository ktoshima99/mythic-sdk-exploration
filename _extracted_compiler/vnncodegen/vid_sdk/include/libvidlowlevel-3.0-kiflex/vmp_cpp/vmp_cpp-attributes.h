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
* FILENAME: vmp_cpp-attributes.h
*
* DESCRIPTION: short versions of attributes
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/


# ifndef __VMP_CPP_ATTRIBUTES_H__
#  define __VMP_CPP_ATTRIBUTES_H__

#ifndef __videantis_lint__
#  define __byteaddress __attribute__((byteaddress))
#  define __wordaddress __attribute__((wordaddress))
#else
#  define __byteaddress
#  define __wordaddress
#endif // __videantis_lint__


# endif // __VMP_CPP_ATTRIBUTES_H__
