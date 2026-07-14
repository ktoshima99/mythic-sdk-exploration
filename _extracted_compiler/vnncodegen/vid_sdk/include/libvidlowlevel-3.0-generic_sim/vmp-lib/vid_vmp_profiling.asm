/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2022 videantis GmbH
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
 * FILENAME: vid_vmp_profiling.asm
 *
 * DESCRIPTION: videantis v-MP assembly profiling macros
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP timer profiling functions
 *
 * @details
 * This file TODO
 *
 * @file vid_vmp_profiling.asm
 */

/// @cond DOXYGEN_IGNORE_ASM



/// Timer control register TIMERCTRL configuration options

.equ VID_TCTRL_DIR_KEEP_VALUE              = 0 // TIMER stopped
.equ VID_TCTRL_DIR_COUNT_UP                = 1 // TIMER count up
.equ VID_TCTRL_DIR_COUNT_DOWN              = 2 // TIMER count down
.equ VID_TCTRL_DIR_COUNT_DOWN_CHANGE_STATE = 3 // TIMER stops when reaching zero and change state

.equ VID_TCTRL_TRIG_RUN                    = 0<<2 // TIMER counts in state RUN
.equ VID_TCTRL_TRIG_RUN_TRIGGER_PC         = 1<<2 // FIXME
.equ VID_TCTRL_TRIG_ALWAYS                 = 2<<2 // TIMER counts every core cycle

// possible states for VID_TCTRL_DIR_COUNT_DOWN_CHANGE_STATE
.equ VID_TCTRL_NEXT_MP_STATE_RUN           = 0<<4 // change to state RUN
.equ VID_TCTRL_NEXT_MP_STATE_HOLD          = 1<<4 // change to state HOLD
.equ VID_TCTRL_NEXT_MP_STATE_SLEEP         = 3<<4 // change to state SLEEP
.equ VID_TCTRL_NEXT_MP_STATE_WAIT          = 5<<4 // change to state WAIT
.equ VID_TCTRL_NEXT_MP_STATE_BREAK         = 4<<4 // change to state BREAK

.equ VID_TCTRL_TRIGGER_PC_SHIFT            =    8 // FIXME

.equ VID_TIMER1 = 1
.equ VID_TIMER2 = 2
.equ VID_TIMER_BOTH = {VID_TIMER1 | VID_TIMER2}




/// Timer profiling function C-interface
.csection vmpprof

.export _vid_prof_start
.export _vid_prof_stop

/// void vid_prof_start(void);
/// Function to stop and initialize both TIMER registers
/// TIMER1 configured to count all cycles
/// TIMER2 configured to count all cycles when in running state (e.g. no DMA wait)
_vid_prof_start:
    // stop TIMER1 and initialize to 0
    MVI	TCTRL1, #VID_TCTRL_DIR_KEEP_VALUE
    MVI	TIMER1, #0

    // stop TIMER2 and initialize to 0
    MVI	TCTRL2, #VID_TCTRL_DIR_KEEP_VALUE
    MVI	TIMER2, #0

    //start TIMER1 counting all core cycles
    MVI	TCTRL1, #{VID_TCTRL_DIR_COUNT_UP | VID_TCTRL_TRIG_ALWAYS}
     //start TIMER2 counting all core cycles when in running state (e.g. no DMA wait)
    MVI	TCTRL2, #{VID_TCTRL_DIR_COUNT_UP | VID_TCTRL_TRIG_RUN}

    JLA     ZERO, lr

/// uint2 vid_prof_stop(void);
/// Function to stop and read both TIMER registers
/// returns two profiling 32Bit values in a uint2 vector word
/// .s1 = all cycles when v-MP was in WAIT state since vmp_prof_start()
/// .s0 = all cycles since vmp_prof_start()
_vid_prof_stop:
    // stop both timers
    MVI	TCTRL1, #VID_TCTRL_DIR_KEEP_VALUE
    MVI	TCTRL2, #VID_TCTRL_DIR_KEEP_VALUE

    MV      mstmp0, TIMER1         //all cycle counts
    MV      mstmp1, TIMER2         //cycle counts excluding DMA waiting
    SUB     mstmp3, mstmp0, mstmp1 //cycle counts for DMA waiting only
    MV      STORE_HIGH, mstmp3
    MV      vparam0, mstmp0       // return uint2   | cyc DMA only | cyc all since _prof_start |
    JLA     ZERO, lr
.endsection




.csection vmprof2

.export _vid_prof_read_timers
.export _vid_prof_write_timers
.export _vid_prof_ctrl_timers
.export _vid_prof_set_timers

/// uint2 vid_prof_read_timers(void);
/// returns two profiling 32Bit values in a uint2 vector word
/// timers are not stopped
/// .s1 = all cycles excluding cycles when v-MP was in WAIT state since vmp_prof_start()
/// .s0 = all cycles since vmp_prof_start()
_vid_prof_read_timers:
    MV      STORE_HIGH, TIMER2
    MV      vparam0, TIMER1       // return uint2   | cycles TIMER2 | cycles TIMER1 |
    JLA     ZERO, lr

/// void vid_prof_write_timers(uint timerSelect, uint2 timerValues);
/// write value to one or both timer register
.equ timerval1 = SR2
.equ timerval2 = SR3
_vid_prof_write_timers:
    MV    timerval1, vparam0
    MV    timerval2, LOAD_HIGH
    MVI     SCONDSEL, #COND_NZ
    ANDICS   ZERO, sparam0, #VID_TIMER1
    MVCR	TIMER1 , timerval1
    ANDICS   ZERO, sparam0, #VID_TIMER2
    MVCR	TIMER2, timerval2
    JLA     ZERO, lr

/// void vid_prof_ctrl_timers(uint timerSelect, uint tctrl1, uint tctrl2);
/// set timer control register of one or both timers
_vid_prof_ctrl_timers:
    MVI     SCONDSEL, #COND_NZ
    ANDICS   ZERO, sparam0, #VID_TIMER1
    MVCR	TCTRL1, sparam1
    ANDICS   ZERO, sparam0, #VID_TIMER2
    MVCR	TCTRL2, sparam2
    JLA     ZERO, lr


/// void vid_prof_set_timers(uint2 timerValues, uint2 timerCtrl);
_vid_prof_set_timers:
    // stop both timers
    MVI	TCTRL1, #VID_TCTRL_DIR_KEEP_VALUE
    MVI	TCTRL2, #VID_TCTRL_DIR_KEEP_VALUE

    // set start values
    MV	TIMER1, vparam0
    MV	TIMER2, LOAD_HIGH

    // set timer configurations
    MV	TCTRL1, vparam1
    MV	TCTRL2, LOAD_HIGH

    JLA     ZERO, lr

.endsection


/// @endcond
