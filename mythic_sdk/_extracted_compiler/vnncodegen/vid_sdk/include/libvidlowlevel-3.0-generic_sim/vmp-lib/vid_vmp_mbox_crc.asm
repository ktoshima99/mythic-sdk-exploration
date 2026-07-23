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
 * FILENAME: vid_vmp_mbox_crc.asm
 *
 * DESCRIPTION: videantis v-MP mailbox interface library with crc
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis v-MP mailbox interface library with crc
 *
 * @details
 * This file implements assembler functions for
 * all videantis v-MP mailbox interface library functions
 * with crc
 *
 * @file vid_vmp_mbox_crc.asm
 */

/// @cond DOXYGEN_IGNORE_ASM

/**
 * @fn int vid_crc_rcv(__local vid_mbox_t* msg_ptr)
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
 * @param[out] msg_ptr Pointer to vid_mbox_t structure, defining storage area for the received message payload
 * @return crc status
 */
.csection vid_crc_rcv
.export _vid_crc_rcv

_vid_crc_rcv:
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
    V_ADD_U32   mvtmp0, (vidx1 @dmem2), mvtmp1

    /* address of message now in mvtmp0 */
    /* DMA from external memory to internal VLL_MSG_IN */
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    /* load message to mbox_ptr */
    MV      vidx0, sparam0

    //vmp_dma_read(_dst, _src, _width, _channel)
    .call   vmp_dma_read(sparam0, mvtmp0, #mbox.VLL_MSG_MAX_SIZE, #mbox.VLL_MSG_DMA_CHANNEL)
    .call   vmp_wait_channel(mbox.VLL_MSG_DMA_CHANNEL)

    /* initialize crc */
    V_MVI_8 mvtmp5, #0x0
    MVI     VCONDSEL, #COND_Z
    V_MVI_8 mvtmp1, #0x80
    V_MVI_8 mvtmp2, #0xd8
    MVI     mstmp0, #mbox.VLL_MSG_PAYLOAD_SIZE

    /* calculate crc */
payload_loop:
    // remainder ^= msg_ptr->payload.pl_ui8[word];
    V_XOR_8     mvtmp5, (vidx0 @dmem)+, mvtmp5
.loop=[0..7]
    // remainder = select(remainder << set_uchar8(1u), (remainder << set_uchar8(1u)) ^ set_uchar8(POLYNOMIAL), (remainder & set_uchar8(TOPBIT)) == set_uchar8(TOPBIT));
    V_AND_8     mvtmp3, mvtmp5, mvtmp1
    V_SLI_8     mvtmp5, mvtmp5, #0x1
    V_XOR_8     mvtmp4, mvtmp5, mvtmp2
    V_SUBCS_8   VZERO, mvtmp3, mvtmp1
    v_MVCR_8    mvtmp5, mvtmp4
.endloop

    ELOOPR      mstmp0, payload_loop

    /* compare calculated crc against received crc */
    V_MVI_32    mvtmp0, #mbox.VLL_ERR
    V_SUBCS_32  VZERO, (vidx0 @dmem), mvtmp5
    V_MVICR_32  mvtmp0, #mbox.VLL_OK
    V_SRI_64    mvtmp1, mvtmp0, #32
    V_OR_64     mvtmp0, mvtmp1, mvtmp0
    MV          sparam0, mvtmp0

    JLA ZERO, lr
.endsection


/**
 * @fn void vid_crc_send(__local vid_mbox_t* msg_ptr)
 * @brief Send a message to another core
 *
 * The function assembles a message from the payload and tries to place it in the
 * incoming mailbox of the receiving core. The receiving core is implicitly defined by
 * the argument of the mailbox initialization function call vid_mbox_init().
 * The operation works in blocking-mode, only. The
 * function returns after it has successfully delivered the message.
 *
 * @param[in] msg_ptr Pointer to vid_mbox_t structure, defining a message
 */
.csection vid_crc_send
.export _vid_crc_send

_vid_crc_send:
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
    V_ADD_U32   mvtmp0, (vidx1 @dmem2), mvtmp0

    /* address now in mvtmp0 */
    MV      vidx0, sparam0

    /* initialize crc */
    V_MVI_8 mvtmp5, #0x0
    MVI     VCONDSEL, #COND_Z
    V_MVI_8 mvtmp1, #0x80
    V_MVI_8 mvtmp2, #0xd8
    MVI     mstmp0, #mbox.VLL_MSG_PAYLOAD_SIZE

    /* calculate crc */
payload_loop:
    // remainder ^= msg_ptr->payload.pl_ui8[word];
    V_XOR_8     mvtmp5, (vidx0 @dmem)+, mvtmp5
.loop=[0..7]
    // remainder = select(remainder << set_uchar8(1u), (remainder << set_uchar8(1u)) ^ set_uchar8(POLYNOMIAL), (remainder & set_uchar8(TOPBIT)) == set_uchar8(TOPBIT));
    V_AND_8     mvtmp3, mvtmp5, mvtmp1
    V_SLI_8     mvtmp5, mvtmp5, #0x1
    V_XOR_8     mvtmp4, mvtmp5, mvtmp2
    V_SUBCS_8   VZERO, mvtmp3, mvtmp1
    v_MVCR_8    mvtmp5, mvtmp4
.endloop

    ELOOPR  mstmp0, payload_loop

    /* store crc result in flag and message type data field*/
    V_STORE (vidx0 @dmem), mvtmp5

    /* send message including crc */
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
    MV  mstmp0, mbox.VLL_MSG_WHOAMI
    SRICS mstmp0, mstmp0, #31
    MVICR IRQOUT, #1


    JLA ZERO, lr
.endsection

/// @endcond
