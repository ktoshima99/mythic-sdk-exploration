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
 * FILENAME: vid_vmp_mbox.asm
 *
 * DESCRIPTION: videantis v-MP mailbox interface library
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP mailbox interface library
 *
 * @details
 * This file implements assembler functions for
 * all videantis v-MP mailbox interface library functions
 *
 * @file vid_vmp_mbox.asm
 */

/// @cond DOXYGEN_IGNORE_ASM
/* ========================================================================== *
 * Mailbox data section                                                       *
 * ========================================================================== */

// check if memory for mbox dsection is defined (default dmem2)
.if !defined(VMP_MBOX_MEM)
.equ VMP_MBOX_MEM = 2
.endif

// translate mbox dsection location
.if {VMP_MBOX_MEM == 1}
.equ MBOX_MEM = dmem
.elseif {VMP_MBOX_MEM == 3}
.equ MBOX_MEM = dmem3
.else
.equ MBOX_MEM = dmem2
.endif

.dsection mbox, MBOX_MEM
.org auto

.equ VLL_MSG_VALID_MASK    = 1
.equ VLL_MSG_MSGTYPE_VALID = 0

.equ VLL_MSG_MAX_SIZE           = {64/8}
.equ VLL_MSG_MAX_SIZE_BYTES     = 64
.equ VLL_MSG_MAX_SIZE_LD        = 6  //1<<6 = 64
.equ VLL_MSG_PAYLOAD_SIZE       = {56/8}
.equ VLL_MSG_PAYLOAD_SIZE_BYTES = 56
.equ VLL_READY_MAX_SIZE         = {8/8}
.equ VLL_READY_MAX_SIZE_BYTES   = 8
.equ VLL_MSG_PER_BOX            = 4
.equ VLL_MSG_MAX_IDX            = {VLL_MSG_PER_BOX - 1}
.equ VLL_MBOX_SIZE              = {VLL_MSG_MAX_SIZE * VLL_MSG_PER_BOX}
.equ VLL_MBOX_SIZE_BYTES        = {{VLL_MSG_MAX_SIZE * VLL_MSG_PER_BOX} << 3}
.equ VLL_MBOX_IDX_MOD           = 16
.equ VLL_MBOX_IDX_MASK          = {VLL_MBOX_IDX_MOD - 1}
.equ VLL_MSG_DMA_CHANNEL        = 15

/*
 * default GPDATA register usage:
 * GPDATA0: High part of 64-bit base mbox address for this core, used until init function called
 * GPDATA1: Low part of 64-bit base mbox address for this core, used until init function called
 * GPDATA2: Core ID, used until init function called
 * GPDATA4: h2c_posted   # of message sent by host/v-SP
 * GPDATA5: h2c_consumed # of messages received by v-MP core
 * GPDATA6: c2h_posted   # of messages sent by v-MP core
 * GPDATA7: c2h_consumed # of messages received by host/v-SP
 */
.equ H2C_POSTED_COUNT   = GPDATA4
.equ H2C_CONSUMED_COUNT = GPDATA5
.equ C2H_POSTED_COUNT   = GPDATA6
.equ C2H_CONSUMED_COUNT = GPDATA7

.equ WAIT_MASK_H2C_POSTED_COUNT   = {{1<<21} | {1<<{GPDATA4 - GPDATA0}}}
.equ WAIT_MASK_C2H_CONSUMED_COUNT = {{1<<21} | {1<<{GPDATA7 - GPDATA0}}}

.equ GPDFLAGS_MASK_H2C_POSTED_COUNT   = {~{1<<{GPDATA4 - GPDATA0}}}
.equ GPDFLAGS_MASK_C2H_CONSUMED_COUNT = {~{1<<{GPDATA7 - GPDATA0}}}

/* identifiers for cores */
.equ VLL_ID_MP_0 =  0
.equ VLL_ID_MP_1 =  1
.equ VLL_ID_MP_2 =  2
.equ VLL_ID_MP_3 =  3
.equ VLL_ID_MP_4 =  4
.equ VLL_ID_MP_5 =  5
.equ VLL_ID_MP_6 =  6
.equ VLL_ID_MP_7 =  7
.equ VLL_ID_SP_0 = 20
.equ VLL_ID_SP_1 = 21
.equ VLL_ID_HOST = 32
.equ VLL_NUM_MP  =  8
.equ VLL_NUM_SP  =  2

    /* total size of data segment = 4 words */
    // external base address of send messages
    .alloc VLL_MSG_EXT_BASEADDR_SND[1]
    // external base address of receive messages
    .alloc VLL_MSG_EXT_BASEADDR_RCV[1]
    // core ID
    .alloc VLL_MSG_WHOAMI[1]
    // temporary memory to save intermediate data
    .alloc tmp[1]

// return codes
.equ VLL_OK  =  0
.equ VLL_ERR = -1

.endsection

/* ============================================================================== *
 * DMA transfer macros used for mailbox send and receive
 * ============================================================================== */
// DMA read transfer
.macro vmp_dma_read(_dst, _src, _width, _channel)
    // write 64Bit external address from vector register _src to vmp_dma_t::extByteaddr
    V_STORE {dma_descr.dma0.start + _channel*dma_descr.CHANNELSIZE}, _src

    // intWordaddr = destination address from scalar register _dst
    MV   STORE_HIGH, _dst
    // length = number of words to read from scalar register _width
    // write to vmp_dma_t::xferFlags_intWordaddr_lengthInt_length
    MVIL {dma_descr.dma0.start + _channel*dma_descr.CHANNELSIZE + 1}, _width    // FIXME: won't work with dmem3 as dma descriptor segment (no direct addressing of dmem3 with MV instruction)

    // 1D transfer count=1 to vmp_dma_t::reserved_strideInt_count3d_count
    MVI     {dma_descr.dma0.start + _channel*dma_descr.CHANNELSIZE + 2}, #1     // FIXME: won't work with dmem3 as dma descriptor segment (no direct addressing of dmem3 with MV instruction)
    // 1D transfer, all strides zero to vmp_dma_t::stride3d_stride
    V_STORE {dma_descr.dma0.start + _channel*dma_descr.CHANNELSIZE + 3}, VZERO

    // trigger DMA read transfer in channel _channel
    .dependency dmem3, 0
    .dependency dmem2, 0
    .dependency dmem, 0
    MVI BIU_DMA_CTRL, #{dma_descr.DMA_PRIO_STD | dma_descr.DMA_READ | _channel}
.endmacro


// DMA write transfer
.macro vmp_dma_write(_dst, _src, _width, _channel)
    // write 64Bit external address from vector register _dst to vmp_dma_t::extByteaddr
    V_STORE {dma_descr.dma0.start + _channel*dma_descr.CHANNELSIZE}, _dst

    // intWordaddr = source address from scalar register _src
    MV   STORE_HIGH, _src
    // length = number of words to read from scalar register _width
    // write to vmp_dma_t::xferFlags_intWordaddr_lengthInt_length
    MVIL {dma_descr.dma0.start + _channel*dma_descr.CHANNELSIZE + 1}, _width    // FIXME: won't work with dmem3 as dma descriptor segment (no direct addressing of dmem3 with MV instruction)

    // 1D transfer count=1 to vmp_dma_t::reserved_strideInt_count3d_count
    MVI     {dma_descr.dma0.start + _channel*dma_descr.CHANNELSIZE + 2}, #1     // FIXME: won't work with dmem3 as dma descriptor segment (no direct addressing of dmem3 with MV instruction)
    // 1D transfer, all strides zero to vmp_dma_t::stride3d_stride
    V_STORE {dma_descr.dma0.start + _channel*dma_descr.CHANNELSIZE + 3}, VZERO

    // trigger DMA write transfer in channel _channel
    .dependency dmem3, 0
    .dependency dmem2, 0
    .dependency dmem, 0
    MVI BIU_DMA_CTRL, #{dma_descr.DMA_PRIO_STD | dma_descr.DMA_WRITE | _channel}
.endmacro

// DMA wait on channel _channel
.macro vmp_wait_channel(_channel)
    WAIT    #{1 << _channel}
.endmacro

/**
 * @fn void vid_mbox_init(__WORDADDRESS __local vid_mbox_t* msg_ptr, uint core_id)
 * @brief Initialize the mailbox for communication with another core
 *
 * The function initializes the mailbox system on a v-MP core for communication with another core. The
 * other core can be the host, or one of the v-SP cores in the system. At a given point of time, a v-MP
 * core can only communicate with one other core via the mailbox system.
 *
 * Valid values for `core_id` are `VLL_ID_HOST`, `VLL_ID_SP_0`, and `VLL_ID_SP_1`.
 *
 * @implements_hlc{LLLIB_HLC_01} @implements_su{LLLIB_SU_01}
 *
 * @param[in] msg_ptr Pointer to vid_mbox_t structure, defining a message
 * @param[in] core_id Core ID of the core to communicate with
 */
.csection vid_mbox_init
.export _vid_mbox_init

_vid_mbox_init:

    /*
     * The core gets the following data passed from the core that
     * initializes the mailboxes:
     * GPDATA0: 64-bit base address of mailbox pair to use (high part)
     * GPDATA1: 64-bit base address of mailbox pair to use (low part)
     * GPDATA2: Core ID
     *
     * The core triggers for GPDATA1 != 0, and then reads GPDATA0 and GPDATA2
     */
l_wait_mbox_config_data:
    MV     mstmp1, GPDATA1
    ADDICS ZERO, mstmp1, #0
    BSSR_PNT l_wait_mbox_config_data, #COND_Z

    /* set up mbox addresses for read and write */
    MV      STORE_HIGH, GPDATA0
    MV      mbox.VLL_MSG_EXT_BASEADDR_SND, mstmp1   // FIXME: won't work with dmem3 as mbox segment (no direct addressing of dmem3 with MV instruction)
    MVI     mstmp0, #mbox.VLL_MBOX_SIZE_BYTES
    ADD     mstmp1, mstmp1, mstmp0
    MV      mbox.VLL_MSG_EXT_BASEADDR_RCV, mstmp1   // FIXME: won't work with dmem3 as mbox segment (no direct addressing of dmem3 with MV instruction)

    MVI SCONDSEL, #COND_NZ
    MVFIRI sidx0, #MP_VERSION
    SRI mstmp2, (sidx0), #VIDASM_PID2
    MVIL mstmp3, #VIDASM_PID1
    SUBCS mstmp2, mstmp2, mstmp3
    // FIXME: the assembler has no SOC for a chip with v-MP 4.0 available
    // Because of this the check will fail and no programm code will be executed
.if{0}
    MVICR GPDATA7, #0x0f
    MVICR lr, #_exit
.endif

    /* save core ID passed in GPDATA2 */
    MVI STORE_HIGH, #0 // make sure to initialize mbox.VLL_MSG_WHOAMI.s1 = 0
    MV  mstmp1, GPDATA2
    MV  mbox.VLL_MSG_WHOAMI, mstmp1     // FIXME: won't work with dmem3 as mbox segment (no direct addressing of dmem3 with MV instruction)

    JLA	ZERO, lr
.endsection

/**
 * @fn uint vid_mbox_rcv(__WORDADDRESS __local vid_mbox_t* msg_ptr)
 * @brief Receive a message from another core
 *
 * The function receives a message from the sending core and places it in the storage provided by the
 * parameter msg_ptr. The sending core is implicitly defined by
 * the argument of the mailbox initialization function call vid_mbox_init().
 * The operation works in blocking-mode, only. The function returns after it
 * has successfully received a message. A received message remains in the mailbox slot and needs to
 * be released with subsequent call to vid_mbox_rel(). Messages need to be received and released
 * in-order by the v-MP.
 *
 * @implements_hlc{LLLIB_HLC_01} @implements_su{LLLIB_SU_02}
 *
 * @param[out] msg_ptr Pointer to vid_mbox_t structure, defining storage area for the received message payload
 * @return Message type value of the message received
 */
.csection vid_mbox_rcv
.export _vid_mbox_rcv

_vid_mbox_rcv:
    // read consumed messages counter in H2C_CONSUMED_COUNT
    MV  mstmp1, mbox.H2C_CONSUMED_COUNT

    JLR ZERO, entry2

    // wait for new messages
l_wait_new_messages:
    WAIT    #mbox.WAIT_MASK_H2C_POSTED_COUNT
    // host has written to H2C_POSTED_COUNT, acknowledge in GPDFLAGS
    MVI     GPDFLAGS, #mbox.GPDFLAGS_MASK_H2C_POSTED_COUNT

    // check if new message is available
entry2:
    MV      mstmp0, mbox.H2C_POSTED_COUNT
    SUBCS   ZERO, mstmp0, mstmp1
    BSSR_PNT l_wait_new_messages, #COND_Z
    // There is at least one message available, load it

    // calculate external address offset to 64Bit base address (32Bit arithemtic)
    MV          mvtmp1, mstmp1
    V_ANDI_64   mvtmp1, mvtmp1, #mbox.VLL_MSG_MAX_IDX
    V_MULIL_U32 mvtmp1, mvtmp1, #mbox.VLL_MSG_MAX_SIZE_BYTES

    // add offset to base address (32Bit arithemtic)
    V_MVFIRI    vidx1, #mbox.VLL_MSG_EXT_BASEADDR_RCV
    V_ADD_U32   mvtmp0, (vidx1 @vreg_dmem_dmem2_dmem3), mvtmp1

    /* address of message now in mvtmp0 */
    /* DMA from external memory to internal VLL_MSG_IN */
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    /* load message to mbox_ptr */
    MV      sidx0, sparam0

    //vmp_dma_read(_dst, _src, _width, _channel)
    .call   vmp_dma_read(sparam0, mvtmp0, #mbox.VLL_MSG_MAX_SIZE, #mbox.VLL_MSG_DMA_CHANNEL)
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    /* message loaded, access message type at end of message */
    ADDFIRI     sidx0, sidx0, #mbox.VLL_MSG_PAYLOAD_SIZE
    MV          mstmp0, (sidx0)     /* access valid and message type */
    MV          mstmp0, LOAD_HIGH   /* return message type       */
    //set return value
    MV  sparam0, mstmp0


    JLA ZERO, lr
.endsection

/**
 * @fn void vid_mbox_rel(__WORDADDRESS __local vid_mbox_t* msg_ptr)
 * @brief Release the last message received from the mailbox system
 *
 * The function releases a messages from the incoming mailbox. The mailbox slot occupied by this
 * message becomes available again for receiving messages. Messages need to be received and
 * released in-order by the v-MP.
 *
 * @implements_hlc{LLLIB_HLC_01} @implements_su{LLLIB_SU_03}
 *
 * @param[in] msg_ptr Pointer to vid_mbox_t structure
 */
.csection vid_mbox_rel
.export _vid_mbox_rel

_vid_mbox_rel:
    // increment counter for consumed messages
    MV      mstmp0, mbox.H2C_CONSUMED_COUNT
    ADDI    mstmp0, mstmp0, #1
    ANDI    mstmp0, mstmp0, #mbox.VLL_MBOX_IDX_MASK
    MV      mbox.H2C_CONSUMED_COUNT, mstmp0

    JLA ZERO, lr
.endsection


/**
 * @fn void vid_mbox_send(__WORDADDRESS __local vid_mbox_t* msg_ptr, uint msg_type)
 * @brief Send a message to another core
 *
 * The function assembles a message from the message type and the payload and tries to place it in the
 * incoming mailbox of the receiving core. The receiving core is implicitly defined by
 * the argument of the mailbox initialization function call vid_mbox_init().
 * The operation works in blocking-mode, only. The
 * function returns after it has successfully delivered the message.
 *
 * @implements_hlc{LLLIB_HLC_01} @implements_su{LLLIB_SU_04}
 *
 * @param[in] msg_ptr Pointer to vid_mbox_t structure, defining a message
 * @param[in] msg_type Message type to be used for the message
 */
.csection vid_mbox_send
.export _vid_mbox_send

_vid_mbox_send:
    // read posted messages counter in mbox.C2H_POSTED_COUNT
    MV  mstmp0, mbox.C2H_POSTED_COUNT
    MVI  SCONDSEL, #COND_N
    JLR ZERO, entry2

    // wait for a free message slot
l_wait_free_slot:
    WAIT    #mbox.WAIT_MASK_C2H_CONSUMED_COUNT
    // host has written to mbox.C2H_CONSUMED_COUNT, acknowledge in GPDFLAGS
    MVI     GPDFLAGS, #mbox.GPDFLAGS_MASK_C2H_CONSUMED_COUNT

    // check for a free message slot
entry2:
    MV      mstmp1, mbox.C2H_CONSUMED_COUNT
    SUBCS   mstmp2, mstmp0, mstmp1
    ADDICR  mstmp2, mstmp2, #mbox.VLL_MBOX_IDX_MOD-1
    ADDICR  mstmp2, mstmp2, #1
    SUBICS   ZERO, mstmp2, #mbox.VLL_MSG_PER_BOX
    BSSR_PNT l_wait_free_slot, #COND_Z
    // There is at least one free slow available

    // calculate external address offset to 64Bit base address (32Bit arithemtic)
    MV          mvtmp0, mstmp0
    V_ANDI_64   mvtmp0, mvtmp0, #mbox.VLL_MSG_MAX_IDX
    V_MULIL_U32 mvtmp0, mvtmp0, #mbox.VLL_MSG_MAX_SIZE_BYTES

    // add offset to base address (32Bit arithemtic)
    V_MVFIRI    vidx1, #mbox.VLL_MSG_EXT_BASEADDR_SND
    V_ADD_U32   mvtmp0, (vidx1 @vreg_dmem_dmem2_dmem3), mvtmp0

    /* address now in mvtmp0 */
    /* send message directly from mbox_ptr */
    MV          vidx0, sparam0
    V_ADDFIRI   vidx0, vidx0, #mbox.VLL_MSG_PAYLOAD_SIZE

    /* and set valid bit and message type */
    MV      STORE_HIGH, sparam1
    MVI     mstmp0, #mbox.VLL_MSG_VALID_MASK
    MV      mvtmp2, mstmp0

    V_STORE (vidx0), mvtmp2

    /* send message including valid flag and message type */
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)
    //vmp_dma_write(_dst, _src, _width, _channel)
    .call   vmp_dma_write(mvtmp0, sparam0, #mbox.VLL_MSG_MAX_SIZE, #mbox.VLL_MSG_DMA_CHANNEL)
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    /* re-read message partly to make sure posted write has completed */
    //vmp_dma_read(_dst, _src, _width, _channel)
    .call   vmp_dma_read(sparam0, mvtmp0, #1, #mbox.VLL_MSG_DMA_CHANNEL)
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    /* we have sent a new message, increment counter */
    /* sequence counter values are 0..15 */
    /* adjust posted messages counter in mbox.C2H_POSTED_COUNT */
    MV      mstmp0, mbox.C2H_POSTED_COUNT
    ADDI    mstmp0, mstmp0, #1
    ANDI    mstmp0, mstmp0, #mbox.VLL_MBOX_IDX_MASK
    MV      mbox.C2H_POSTED_COUNT, mstmp0

    /* set IRQOUT if message was send to host        */
    /* set IRQOUT depending on value of bit 31 of ID */
    /* set IRQOUT to 1 if bit 31 == 0                */
    MVI SCONDSEL, #COND_Z
    MV  mstmp0, mbox.VLL_MSG_WHOAMI     // FIXME: won't work with dmem3 as mbox segment (no direct addressing of dmem3 with MV instruction)
    SRICS mstmp0, mstmp0, #31
    MVICR IRQOUT, #1


    JLA ZERO, lr
.endsection

/**
 * @fn uint vid_get_core_id(void)
 * @brief Provide the hardware ID of the core
 *
 * The function returns the hardware ID of the core where the function is executed. The returned ID is
 * one of the IDs `VLL_ID_MP_0`, `VLL_ID_MP_1`, `VLL_ID_MP_2`, `VLL_ID_MP_3`,
 * `VLL_ID_MP_4`, `VLL_ID_MP_5`, `VLL_ID_MP_6`, `VLL_ID_MP_7`.
 *
 * @implements_hlc{LLLIB_HLC_04} @implements_su{LLLIB_SU_09}
 *
 * @return Hardware core ID of the v-MP core
 */
.csection vid_get_core_id
.export _vid_get_core_id

_vid_get_core_id:
    MV	sparam0, mbox.VLL_MSG_WHOAMI    // FIXME: won't work with dmem3 as mbox segment (no direct addressing of dmem3 with MV instruction)
    /* remove bit 31 */
    SLI     sparam0, sparam0, #1
    SRI     sparam0, sparam0, #1
    JLA ZERO, lr
.endsection

/**
 * @fn int vid_hwsema_getlock(__WORDADDRESS __local vid_hwsema_t *sema)
 * @brief Try to acquire a lock implemented by a hardware semaphore
 *
 * Try to acquire a lock imlemented by a hardware semaphore. If not successful, the lock acquisition
 * must be re-tried.
 *
 * The function returns the following return codes:
 * - `VLL_OK`: Success, the lock was acquired.
 * - `VLL_ERR`: Error condition, the lock was not acquired.
 *
 * @implements_hlc{LLLIB_HLC_03} @implements_su{LLLIB_SU_10}
 *
 * @param[in] sema Semaphore to use for lock acquisition
 * @return return code
 */
.csection vid_hwsema_getlock
.export _vid_hwsema_getlock

_vid_hwsema_getlock:
    // get address of semaphore from semaphore struct vid_hwsema_t
    // @var uint2 vid_hwsema_t::addr
    // 64-bit address of a hardware semaphore (high part s1 and low part s0)
    MV  SFIR0, sparam0
    MV  mvtmp0, (SFIR0)

    // internal address to save semaphore value
    MVIL sparam2, #mbox.tmp

    // wait for channel to become available
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    // call asm macro vmp_dma_read(_dst, _src, _width, _channel)
    .call   vmp_dma_read(sparam2, mvtmp0, #1, #mbox.VLL_MSG_DMA_CHANNEL)
    // wait for end of transfer
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    // semaphore value in upper 32 bits
    MV  mstmp1, mbox.tmp    // FIXME: won't work with dmem3 as mbox segment (no direct addressing of dmem3 with MV instruction)
    MV  mstmp1, LOAD_HIGH

    // default return value
    MVI  sparam0, #mbox.VLL_ERR
    // if semaphore is 0, we got the semaphore lock
    MVI SCONDSEL, #COND_Z
    SUBCS ZERO, ZERO, mstmp1
    // if semaphore was aquired, return VLL_OK
    MVICR sparam0, #mbox.VLL_OK

    JLA	ZERO, lr

.endsection

/**
 * @fn void vid_hwsema_rellock(__WORDADDRESS __local vid_hwsema_t *sema)
 * @brief Release a lock previously acquired
 *
 * Return a lock implemented by a hardware semaphore that was previously acquired.
 *
 * @implements_hlc{LLLIB_HLC_03} @implements_su{LLLIB_SU_11}
 *
 * @param[in] sema Semaphore to use for lock release
 */
.csection vid_hwsema_rellock
.export _vid_hwsema_rellock

_vid_hwsema_rellock:
    // get address of semaphore from semaphore struct vid_hwsema_t
    // @var uint2 vid_hwsema_t::addr
    // 64-bit address of a hardware semaphore (high part s1 and low part s0)
    MV  SFIR0, sparam0
    MV  mvtmp0, (SFIR0)

    // internal address to save semaphore value
    MVIL sparam2, #mbox.tmp

    // wait for channel to become available
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    // call asm macro vmp_dma_write(_dst, _src, _width, _channel)
    .call   vmp_dma_write(mvtmp0, sparam2, #1, #mbox.VLL_MSG_DMA_CHANNEL)
    // wait for end of transfer
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    JLA	ZERO, lr

.endsection

/// @endcond
