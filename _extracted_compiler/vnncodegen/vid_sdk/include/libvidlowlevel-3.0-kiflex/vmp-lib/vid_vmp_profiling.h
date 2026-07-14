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
 *
 */

#ifndef __VID_VMP_PROFILING_H__
#define __VID_VMP_PROFILING_H__

// vector data types
#include "vmp_cl/vmp_cl-types.h"

#ifdef __videantis_lint__
// emulation library for videantis-C
// required for static code analysis with FlexeLint and oclint
#include "vmp_cl/vid_videantis-C.hpp"
#endif

/// Timer control register TIMERCTRL configuration options
#define VID_TCTRL_DIR_KEEP_VALUE               0u // TIMER stopped
#define VID_TCTRL_DIR_COUNT_UP                 1u // TIMER count up
#define VID_TCTRL_DIR_COUNT_DOWN               2u // TIMER count down
#define VID_TCTRL_DIR_COUNT_DOWN_CHANGE_STATE  3u // TIMER stops when reaching zero and change state

#define VID_TCTRL_TRIG_RUN                     (0u << 2u) // TIMER counts in state RUN
#define VID_TCTRL_TRIG_RUN_TRIGGER_PC          (1u << 2u) // FIXME
#define VID_TCTRL_TRIG_ALWAYS                  (2u << 2u) // TIMER counts every core cycle

/// possible states for VID_TCTRL_DIR_COUNT_DOWN_CHANGE_STATE
#define VID_TCTRL_NEXT_MP_STATE_RUN            (0u << 4u) // change to state RUN
#define VID_TCTRL_NEXT_MP_STATE_HOLD           (1u << 4u) // change to state HOLD
#define VID_TCTRL_NEXT_MP_STATE_SLEEP          (3u << 4u) // change to state SLEEP
#define VID_TCTRL_NEXT_MP_STATE_WAIT           (5u << 4u) // change to state WAIT
#define VID_TCTRL_NEXT_MP_STATE_BREAK          (4u << 4u) // change to state BREAK

#define VID_TCTRL_TRIGGER_PC_SHIFT                8u // FIXME

#define VID_TIMER1  1u
#define VID_TIMER2  2u
#define VID_TIMER_BOTH  (VID_TIMER1 | VID_TIMER2)





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


#endif // __VID_VMP_PROFILING_H__
