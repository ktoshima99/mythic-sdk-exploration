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
* FILENAME: vmp_cpp-printf.h
*
* DESCRIPTION: interface to videantis-C vmprintf() in simulator vidsim
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief Interface to videantis-C vmprintf() in simulator vidsim
 *
 * @details
 * This file provides implementations for printing formatted string
 * similar to C style printf() in the vidsim simulation environment.
 * Printing string is not suported on hardware targets.
 * The functions have no side effects when executed on a hardware target.
 *
 * @file
 */


#ifndef __VMP_CPP_PRINTF_H__
#define __VMP_CPP_PRINTF_H__

// attributes
#include "vmp_cpp-attributes.h"

// vector data types
#include "vmp_cpp-types.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cpp_videantis-C.h"
#endif

//lint -save

//lint -e9026 Note 9026: Function-like macro, '?PRINTF_?', defined [MISRA 2012 Directive 4.9, advisory]
//lintReason Lint exceptions introduced to enable C++ template like functions enabled by macros

// avoid language incompatibility between OpenCL-C and C++ assumed by oclint
#ifdef __videantis_lint__
#define VOLATILE
#define PCAST long
#else
#define VOLATILE volatile
#define PCAST int
#endif

/*
 * ****************************************
 * printf functions
 * ****************************************
 */

//(SYSCALL_PRINTF_VAR_HI & SYSCALL_PRINTFVAR_LO)

/// v-MP internal address to store variable values for simulator printf mechanism
#define SYSCALL_PRINTF_VAR __builtin_vmp_convert_wordaddresstobyteaddress(0x238)

/// v-MP internal address to provide string address for simulator printf mechanism
#define SYSCALL_PRINTF_STR_DMEM                                                \
  __builtin_vmp_convert_wordaddresstobyteaddress(0x23b)

/// convert string internal byte address to word address
#define SYSCALL_PRINTF_STR_POINTER(_byteaddress)                               \
  __builtin_vmp_convert_byteaddresstowordaddress((PCAST)_byteaddress)

/** @brief Macro to generate vmprintf() functions with single value vector argument. */
#define MPRINTF_1(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, single argument value */    \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a);                           \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a) {                          \
    VOLATILE int2 *var_ptr = (VOLATILE int2 *)SYSCALL_PRINTF_VAR;              \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = as_int2(a);                                                     \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 2 vector arguments. */
#define MPRINTF_2(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, two argument values */      \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b);                 \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b) {                \
    VOLATILE int2 *var_ptr = (VOLATILE int2 *)SYSCALL_PRINTF_VAR;              \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = as_int2(a);                                                     \
    *var_ptr = as_int2(b);                                                     \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 3 vector arguments. */
#define MPRINTF_3(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, 3 argument values */        \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  /** @param c Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c);       \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c) {      \
    VOLATILE int2 *var_ptr = (VOLATILE int2 *)SYSCALL_PRINTF_VAR;              \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = as_int2(a);                                                     \
    *var_ptr = as_int2(b);                                                     \
    *var_ptr = as_int2(c);                                                     \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 4 vector arguments. */
#define MPRINTF_4(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, 4 argument values */        \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  /** @param c Argument value of type __TYPE */                                \
  /** @param d Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d);                                                               \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d) {                                                              \
    VOLATILE int2 *var_ptr = (VOLATILE int2 *)SYSCALL_PRINTF_VAR;              \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = as_int2(a);                                                     \
    *var_ptr = as_int2(b);                                                     \
    *var_ptr = as_int2(c);                                                     \
    *var_ptr = as_int2(d);                                                     \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 5 vector arguments. */
#define MPRINTF_5(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, 5 argument values */        \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  /** @param c Argument value of type __TYPE */                                \
  /** @param d Argument value of type __TYPE */                                \
  /** @param e Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d, __TYPE e);                                                     \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d, __TYPE e) {                                                    \
    VOLATILE int2 *var_ptr = (VOLATILE int2 *)SYSCALL_PRINTF_VAR;              \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = as_int2(a);                                                     \
    *var_ptr = as_int2(b);                                                     \
    *var_ptr = as_int2(c);                                                     \
    *var_ptr = as_int2(d);                                                     \
    *var_ptr = as_int2(e);                                                     \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 6 vector arguments. */
#define MPRINTF_6(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, 6 argument values */        \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  /** @param c Argument value of type __TYPE */                                \
  /** @param d Argument value of type __TYPE */                                \
  /** @param e Argument value of type __TYPE */                                \
  /** @param f Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d, __TYPE e, __TYPE f);                                           \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d, __TYPE e, __TYPE f) {                                          \
    VOLATILE int2 *var_ptr = (VOLATILE int2 *)SYSCALL_PRINTF_VAR;              \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = as_int2(a);                                                     \
    *var_ptr = as_int2(b);                                                     \
    *var_ptr = as_int2(c);                                                     \
    *var_ptr = as_int2(d);                                                     \
    *var_ptr = as_int2(e);                                                     \
    *var_ptr = as_int2(f);                                                     \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with single scalar argument. */
#define SPRINTF_1(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, single argument value */    \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a);                           \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a) {                          \
    VOLATILE int *var_ptr = (VOLATILE int *)SYSCALL_PRINTF_VAR;                \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = (int)(a);                                                       \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 2 scalar arguments. */
#define SPRINTF_2(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, two argument values */      \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b);                 \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b) {                \
    VOLATILE int *var_ptr = (VOLATILE int *)SYSCALL_PRINTF_VAR;                \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = (int)(a);                                                       \
    *var_ptr = (int)(b);                                                       \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 3 scalar arguments. */
#define SPRINTF_3(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, 3 argument values */        \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  /** @param c Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c);       \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c) {      \
    VOLATILE int *var_ptr = (VOLATILE int *)SYSCALL_PRINTF_VAR;                \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = (int)(a);                                                       \
    *var_ptr = (int)(b);                                                       \
    *var_ptr = (int)(c);                                                       \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 4 scalar arguments. */
#define SPRINTF_4(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, 4 argument values */        \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  /** @param c Argument value of type __TYPE */                                \
  /** @param d Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d);                                                               \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d) {                                                              \
    VOLATILE int *var_ptr = (VOLATILE int *)SYSCALL_PRINTF_VAR;                \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = (int)(a);                                                       \
    *var_ptr = (int)(b);                                                       \
    *var_ptr = (int)(c);                                                       \
    *var_ptr = (int)(d);                                                       \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 5 scalar arguments. */
#define SPRINTF_5(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, 5 argument values */        \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  /** @param c Argument value of type __TYPE */                                \
  /** @param d Argument value of type __TYPE */                                \
  /** @param e Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d, __TYPE e);                                                     \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d, __TYPE e) {                                                    \
    VOLATILE int *var_ptr = (VOLATILE int *)SYSCALL_PRINTF_VAR;                \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = (int)(a);                                                       \
    *var_ptr = (int)(b);                                                       \
    *var_ptr = (int)(c);                                                       \
    *var_ptr = (int)(d);                                                       \
    *var_ptr = (int)(e);                                                       \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Macro to generate vmprintf() functions with 6 scalar arguments. */
#define SPRINTF_6(__TYPE)                                                      \
  /** @brief Print formatted string in simulation, 6 argument values */        \
  /** @param str Pointer to string with format tags in local data memory */    \
  /** @param a Argument value of type __TYPE */                                \
  /** @param b Argument value of type __TYPE */                                \
  /** @param c Argument value of type __TYPE */                                \
  /** @param d Argument value of type __TYPE */                                \
  /** @param e Argument value of type __TYPE */                                \
  /** @param f Argument value of type __TYPE */                                \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d, __TYPE e, __TYPE f);                                           \
  __attribute__((overloadable, always_inline)) inline void vmprintf(           \
      __byteaddress __constant char *str, __TYPE a, __TYPE b, __TYPE c,        \
      __TYPE d, __TYPE e, __TYPE f) {                                          \
    VOLATILE int *var_ptr = (VOLATILE int *)SYSCALL_PRINTF_VAR;                \
    VOLATILE int *str_ptr = (VOLATILE int *)SYSCALL_PRINTF_STR_DMEM;           \
    *var_ptr = (int)(a);                                                       \
    *var_ptr = (int)(b);                                                       \
    *var_ptr = (int)(c);                                                       \
    *var_ptr = (int)(d);                                                       \
    *var_ptr = (int)(e);                                                       \
    *var_ptr = (int)(f);                                                       \
    *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);                  \
  }

/** @brief Print string in simulation */
/** @param str Pointer to string in local data memory */
__attribute__((overloadable, always_inline)) inline void
vmprintf(__byteaddress __constant char *str);
__attribute__((overloadable, always_inline)) inline void
vmprintf(__byteaddress __constant char *str) {
  VOLATILE int *str_ptr = (VOLATILE int *) SYSCALL_PRINTF_STR_DMEM;
  *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);
}

/** @brief Print string in simulation */
/** @param str Pointer to string in local data memory */
__attribute__((overloadable, always_inline)) inline void
vmprintf(__byteaddress char *str);
__attribute__((overloadable, always_inline)) inline void
vmprintf(__byteaddress char *str) {
  VOLATILE int *str_ptr = (VOLATILE int *) SYSCALL_PRINTF_STR_DMEM;
  *str_ptr = (VOLATILE int)SYSCALL_PRINTF_STR_POINTER(str);
}




SPRINTF_1(int)
SPRINTF_2(int)
SPRINTF_3(int)
SPRINTF_4(int)
SPRINTF_5(int)
SPRINTF_6(int)

SPRINTF_1(uint)
SPRINTF_2(uint)
SPRINTF_3(uint)
SPRINTF_4(uint)
SPRINTF_5(uint)
SPRINTF_6(uint)

MPRINTF_1(long1)
MPRINTF_2(long1)
MPRINTF_3(long1)
MPRINTF_4(long1)
MPRINTF_5(long1)
MPRINTF_6(long1)

MPRINTF_1(ulong1)
MPRINTF_2(ulong1)
MPRINTF_3(ulong1)
MPRINTF_4(ulong1)
MPRINTF_5(ulong1)
MPRINTF_6(ulong1)

MPRINTF_1(int2)
MPRINTF_2(int2)
MPRINTF_3(int2)
MPRINTF_4(int2)
MPRINTF_5(int2)
MPRINTF_6(int2)

MPRINTF_1(uint2)
MPRINTF_2(uint2)
MPRINTF_3(uint2)
MPRINTF_4(uint2)
MPRINTF_5(uint2)
MPRINTF_6(uint2)

MPRINTF_1(short4)
MPRINTF_2(short4)
MPRINTF_3(short4)
MPRINTF_4(short4)
MPRINTF_5(short4)
MPRINTF_6(short4)

MPRINTF_1(ushort4)
MPRINTF_2(ushort4)
MPRINTF_3(ushort4)
MPRINTF_4(ushort4)
MPRINTF_5(ushort4)
MPRINTF_6(ushort4)

MPRINTF_1(char8)
MPRINTF_2(char8)
MPRINTF_3(char8)
MPRINTF_4(char8)
MPRINTF_5(char8)
MPRINTF_6(char8)

MPRINTF_1(uchar8)
MPRINTF_2(uchar8)
MPRINTF_3(uchar8)
MPRINTF_4(uchar8)
MPRINTF_5(uchar8)
MPRINTF_6(uchar8)

//lint -restore

#endif // __VMP_CPP_PRINTF_H__
