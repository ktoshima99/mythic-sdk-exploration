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
 * FILENAME: vid_vmp_profiling.h
 *
 * DESCRIPTION: videantis-C v-MP profiling function declarations
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @file
 * @brief videantis-C v-MP profiling function declarations
 *
 * This file provides declarations for all hardware timer based
 * profiling functions implemented in vid_vmp_profiling.asm
 * and templates for inline access.
 *
 */

#ifndef VID_VMP_CPP_PROFILING_H
#define VID_VMP_CPP_PROFILING_H

// vector data types
#include "vmp_cpp-types.h"
#include "vid_vmp_ioregs.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cpp_videantis-C.h"
#endif

/// Timer control register TIMERCTRL configuration options
enum class TctrlDir : unsigned
{
    KEEP_VALUE              = 0u, ///< TIMER stopped
    COUNT_UP                = 1u, ///< TIMER count up
    COUNT_DOWN              = 2u, ///< TIMER count down
    COUNT_DOWN_CHANGE_STATE = 3u, ///< TIMER stops at zero and change state
};

/// Timer trigger condition
enum class TctrlTrig : unsigned
{
    RUN            = 0u << 2, ///< TIMER counts in state RUN
    RUN_TRIGGER_PC = 1u << 2, ///< FIXME
    ALWAYS         = 2u << 2, ///< TIMER counts every core cycle
};

/// possible next states for VID_TCTRL_DIR_COUNT_DOWN_CHANGE_STATE
enum class TctrlNextState : unsigned
{
    RUN   = 0u << 4, ///< change to state RUN
    HOLD  = 1u << 4, ///< change to state HOLD
    SLEEP = 3u << 4, ///< change to state SLEEP
    WAIT  = 5u << 4, ///< change to state WAIT
    BREAK = 4u << 4, ///< change to state BREAK
};

#define VID_TCTRL_TRIGGER_PC_SHIFT                8u // FIXME

#define VID_TIMER1  1u
#define VID_TIMER2  2u
#define VID_TIMER_BOTH  (VID_TIMER1 | VID_TIMER2)

/// @brief Struct for inline access of TIMERx/TCTRLx registers
/// @tparam TimerAddr TIMERx register address
/// @tparam TctrlAddr TCTRLx register address
template <addr_t TimerAddr, addr_t TctrlAddr>
struct timer_reg
{
    static constexpr addr_t timer_addr  = TimerAddr;
    static constexpr addr_t tctrl_addr = TctrlAddr;

    // read/write timer value
    static inline unsigned get()   { return *(reg32*)timer_addr; }
    static inline void set(unsigned v) { *(reg32*)timer_addr = v; }

    // read/write control register
    static inline unsigned get_ctrl() { return *(reg32*)tctrl_addr; }
    static inline void set_ctrl(unsigned v) { *(reg32*)tctrl_addr = v; }

    // configure timer using enums
    static inline void configure(TctrlDir dir, TctrlTrig trig, TctrlNextState next_state = TctrlNextState::RUN)
    {
        set_ctrl(static_cast<unsigned>(dir) | static_cast<unsigned>(trig) | static_cast<unsigned>(next_state));
    }

    // convenience helpers
    static inline void stop() { configure(TctrlDir::KEEP_VALUE, TctrlTrig::RUN); }
    static inline void start_count_up() { configure(TctrlDir::COUNT_UP, TctrlTrig::RUN); }
    static inline void start_count_down() { configure(TctrlDir::COUNT_DOWN, TctrlTrig::RUN); }
    static inline void start_count_down_change_state(TctrlNextState state) { configure(TctrlDir::COUNT_DOWN_CHANGE_STATE, TctrlTrig::RUN, state); }
};

/// @brief First v-MP hardware timer
using TIMER1 = timer_reg<VMP_BADDR_TIMER1, VMP_BADDR_TCTRL1>;
/// @brief Second v-MP hardware timer
using TIMER2 = timer_reg<VMP_BADDR_TIMER2, VMP_BADDR_TCTRL2>;


#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Function to stop and initialize both TIMER registers
 *
 * TIMER1 configured to count all cycles\n
 * TIMER2 configured to count all cycles when in running state (e.g. no DMA wait)
 */
void vid_prof_start(void);


/**
 * @brief Function to stop and read both TIMER registers
 *
 * returns two profiling 32Bit values in a uint2 vector word
 * .s1 = all cycles when v-MP was in WAIT state since vmp_prof_start()
 * .s0 = all cycles since vmp_prof_start()
 *
 * @return vector uint2 (.s1 = all cycles when v-MP was in WAIT state, .s0 = all cycles
 */
uint2 vid_prof_stop(void);


/**
 * @brief Function to stop and read both TIMER registers
 *
 * returns two profiling 32Bit values in a uint2 vector word
 * timers are not stopped
 * .s1 = all cycles when v-MP was in WAIT state since vmp_prof_start()
 * .s0 = all cycles since vmp_prof_start()
 *
 * @return vector uint2 (.s1 = all cycles when v-MP was in WAIT state, .s0 = all cycles
 */
uint2 vid_prof_read_timers(void);


/**
 * @brief Function to write values into TIMER registers
 *
 * This function writes the provided values into the selected timer registers
 * TIMER1, TIMER2, or both
 *
 * @param[in] timerSelect Defines which timer registers are updated (valid values: VID_TIMER1, VID_TIMER2, VID_TIMER_BOTH)
 * @param[in] vatimerValueslue .s1 = value for TIMER2, .s0 = value for TIMER1
 */
void vid_prof_write_timers(uint timerSelect, uint2 timerValues);


/**
 * @brief Function to write values into TCTRLn registers
 *
 * This function writes the provided values into the selected
 * timer control registers TCTRL1, TCTRL2, or both
 *
 * @param[in] timerSelect Defines which control registers are updated (valid values: VID_TIMER1, VID_TIMER2, VID_TIMER_BOTH)
 * @param[in] tcrl1 configuration for TIMER1 (valid value bit patterns: VID_TCTRL_*)
 * @param[in] tcrl2 configuration for TIMER2 (valid value bit patterns: VID_TCTRL_*)
 */
void vid_prof_ctrl_timers(uint timerSelect, uint tctrl1, uint tctrl2);


/**
 * @brief Function to stop timers and write values into TIMERn and TCTRLn registers
 *
 * This function stops both timers and writes the provided values into
 * timer value registers TIMER1, TIMER2, and
 * timer control registers TCTRL1, TCTRL2
 *
 * @param[in] timerSelect Defines which control registers are updated (valid values: VID_TIMER1, VID_TIMER2, VID_TIMER_BOTH)
 * @param[in] timerCtrl configuration for TIMER2 (.s1) and TIMER1 (.s0) (valid value bit patterns: VID_TCTRL_*)
 */
void vid_prof_set_timers(uint2 timerValues, uint2 timerCtrl);

#ifdef __cplusplus
}
#endif

#endif // VID_VMP_PROFILING_H
