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
 * FILENAME: vid_vmp_registers.h
 *
 * DESCRIPTION: videantis-C v-MP register declarations
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @file
 * @brief videantis-C v-MP register declarations
 *
 * This file provides declarations for all hardware registers
 * and templates for inline access.
 *
 */

#ifndef VID_VMP_CPP_REGISTERS_H
#define VID_VMP_CPP_REGISTERS_H

// vector data types
#include "vmp_cpp-types.h"
#include "vid_vmp_ioregs.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cpp_videantis-C.h"
#endif

/// @brief Template class representing GPDATA registers
/// @tparam Bit GPDATAn index n
template <unsigned Bit>
struct gpdata_reg
{
    static_assert(Bit < 8, "GPDATA bit must be 0..7");

    static constexpr addr_t addr = VMP_BADDR_GPDATA0 + (Bit << 3u);

    /// @brief read GPDATA register
    /// @return value read from GPDATA
    static inline unsigned get() { return *(reg32 *)addr; }

    /// @brief write GPDATA register
    /// @param v value to be written to GPDATA
    static inline void set(unsigned v) { *(reg32 *)addr = v; }

    /// @brief Acknowledge GPDATA register after WAIT on register value written from other processor
    static inline void ack() { *(reg32 *)(VMP_BADDR_GPDFLAGS) = ~(1u << Bit); }

    /// @brief WAIT until value is written to this GPDATA triggers from other processor
    /// @param acknowledge boolean parameter to acknowledge data and enable next WAIT
    /// @return value read from GPDATA
    static inline unsigned wait(bool acknowledge = false)
    {
        __builtin_vmp_wait(VMP_WAIT_MODE_GPDATA | (1u << Bit));
        unsigned value = *(reg32 *)addr;
        if (acknowledge)
            ack();
        return value;
    }

    /// @brief Read modify write GPDATA set bits
    /// @param mask Bits to be set masked with GPDATA content
    static inline void set_bits(unsigned mask) { *(reg32 *)addr |= mask; }

    /// @brief Read modify write GPDATA clear bits
    /// @param mask Bits to be cleared masked with GPDATA content
    static inline void clear_bits(unsigned mask) { *(reg32 *)addr &= ~mask; }
};

using GPDATA0 = gpdata_reg<0>;
using GPDATA1 = gpdata_reg<1>;
using GPDATA2 = gpdata_reg<2>;
using GPDATA3 = gpdata_reg<3>;
using GPDATA4 = gpdata_reg<4>;
using GPDATA5 = gpdata_reg<5>;
using GPDATA6 = gpdata_reg<6>;
using GPDATA7 = gpdata_reg<7>;



template <unsigned WaitMask, addr_t Addr>
struct irq_reg_t
{
    static constexpr addr_t addr = Addr;
    static constexpr unsigned mask = WaitMask;

    static inline unsigned get() { return *(reg32*)addr; }
    static inline void set(unsigned v) { *(reg32*)addr = v; }
    static inline void wait() { __builtin_vmp_wait(mask); }
};


using IRQIC  = irq_reg_t<0,  VMP_ADDR_IRQIC>; /* no wait mask */
using IRQIN  = irq_reg_t<VMP_WAIT_IRQIN_SET,  VMP_ADDR_IRQIN>;
using IRQOUT_SET   = irq_reg_t<VMP_WAIT_IRQOUT_SET,  VMP_ADDR_IRQOUT>;
using IRQOUT_CLEAR = irq_reg_t<VMP_WAIT_IRQOUT_CLEAR, VMP_ADDR_IRQOUT>;


using BOOT_ADDR_L = irq_reg_t<0, VMP_BADDR_BOOT_ADDR_L>;
using BOOT_ADDR_H = irq_reg_t<0, VMP_BADDR_BOOT_ADDR_H>;

inline unsigned long long boot_addr_get()
{
    return ( (unsigned long long)  (BOOT_ADDR_H::get()) << 32) | BOOT_ADDR_L::get();
}

#endif
