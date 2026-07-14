/*++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++

 CONFIDENTIAL AND PROPRIETARY INFORMATION
 Copyright 2004 - 2026 videantis GmbH
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
 * FILENAME: vid_lowlevelif_vmp.h
 *
 * DESCRIPTION: videantis low level if API include file for v-MP
 *
 *++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

/**
 * @brief videantis lowlevel interface API include file for v-MP
 *
 * @details
 * This file contains all macros definitions and function prototypes
 * to use the videantis Lowlevel Library.
 *
 * @file vid_lowlevelif_vmp.h
 */

//lint -save

#ifndef VID_LOWLEVELIF_VMP_H
#define VID_LOWLEVELIF_VMP_H

// llvm interface to videantis-C
#ifdef __cplusplus
#include "vmp_cpp/vmp4_cpp.h"
#else
#include "vmp4_cl.h"
#endif


// videantis LowLevel Library common include file
#include "vid_lowlevelif_common.h"

#ifdef __cplusplus
extern "C" {
#endif
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
 *
 * @startuml{vid_mbox_init.png}
 * skinparam defaultTextAlignment center
 * start
 * :wait until GPDATA0 is not zero;
 * :set up mbox addresses for read and write;
 * :save core ID passed in GPDATA1;
 * :initialize send message counter;
 * stop
 * @enduml
 *
 * ## Dynamic behaviour
 *
 * @startuml{vid_mbox_init_dynamic.png}
 * participant HOST
 * participant Core
 *
 * rnote over Core
 * Wait for mbox init
 * from host
 * endrnote
 *
 * HOST -> Core ++ : Init_Mbox
 *
 * rnote over Core
 * Set up mbox addresses,
 * save core ID and
 * init send msg counter
 * endrnote
 * @enduml
 */
void vid_mbox_init(__WORDADDRESS __local vid_mbox_t* msg_ptr, uint core_id);

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
 *
 * @startuml{vid_mbox_send.png}
 * skinparam defaultTextAlignment center
 * start
 * :store registers used on stack;
 * :check message slot availability;
 * if (no message slot available?) then (yes)
 *   :wait for slot to save message;
 * else (no)
 * endif
 * :set valid bit and message type;
 * :write message to free slot;
 * :adjust posted messages counter;
 * if (bit 31 of core ID is 0) then (yes)
 *   :raise IRQOUT;
 * else (no)
 * endif
 * :restore used registers from stack;
 * stop
 * @enduml
 *
 * ## Dynamic behaviour
 *
 * @startuml{vid_mbox_send_dynamic.png}
  * participant HOST
 * participant Core
 *
 * rnote over Core
 * Wait for free
 * message slot
 * endrnote
 *
 * HOST -> Core ++ : Consume_Msg
 *
 * rnote over Core
 * Prepare message,
 * write message to free slot and
 * increment posted message counter
 * endrnote
 * @enduml
 */
void vid_mbox_send(__WORDADDRESS __local vid_mbox_t* msg_ptr, uint msg_type);

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
 *
 * @startuml{vid_mbox_rcv.png}
 * skinparam defaultTextAlignment center
 * start
 * :save registers used on stack;
 * :check for new messages;
 * if (no new message available?) then (yes)
 *   :wait until new message is available;
 * else (no)
 * endif
 * :load message from external memory;
 * :set return value;
 * :restore used registers from stack;
 * stop
 * @enduml
 *
 * ## Dynamic behaviour
 *
 * @startuml{vid_mbox_rcv_dynamic.png}
 * participant HOST
 * participant Core
 *
 * rnote over Core
 * Wait for new mbox
 * message from host
 * endrnote
 *
 * HOST -> Core ++ : Send_Msg
 *
 * rnote over Core
 * Load message and
 * return message type
 * endrnote
 * @enduml
 */
uint vid_mbox_rcv(__WORDADDRESS __local vid_mbox_t* msg_ptr);

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
 *
 * @startuml{vid_mbox_rel.png}
 * skinparam defaultTextAlignment center
 * start
 * :increment counter for consumed messages;
 * stop
 * @enduml
 */
void vid_mbox_rel(__WORDADDRESS __local vid_mbox_t* msg_ptr);

/**
 * @fn void vid_crc_send(__WORDADDRESS __local vid_mbox_t* msg_ptr)
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
void vid_crc_send(__WORDADDRESS __local vid_mbox_t* msg_ptr);

/**
 * @fn int vid_crc_rcv(__WORDADDRESS __local vid_mbox_t* msg_ptr)
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
int vid_crc_rcv(__WORDADDRESS __local vid_mbox_t* msg_ptr);

/**
 * @fn int vid_hwsema_getlock(__local vid_hwsema_t *sema)
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
 *
 * @startuml{vid_hwsema_getlock.png}
 * skinparam defaultTextAlignment center
 * start
 * :save compiler-used registers;
 * :get address of semaphore from semaphore id struct;
 * :wait for channel to become available;
 * :read semaphore to lock it;
 * :wait for read to finish;
 * :generate return value;
 * :restore compiler-used registers;
 * stop
 * @enduml
 *
 * ## Dynamic behaviour
 *
 * @startuml{vid_hwsema_dynamic.png}
 * participant Core_0
 * participant Core_1
 * participant HW_Semaphore
 *
 * Core_0 -> HW_Semaphore : Get_Semaphore
 * activate HW_Semaphore
 * activate Core_0
 *
 * Core_1 -> HW_Semaphore : Get_Semaphore
 * activate Core_1
 * rnote over Core_1
 * Get_Semaphore
 * not successful
 * endrnote
 * deactivate Core_1
 *
 *
 * rnote over Core_0
 * Safe to operate on
 * shared resource
 * because of owning the
 * hardware semaphore
 * endrnote
 *
 * Core_0 -> HW_Semaphore : Release_Semaphore
 * deactivate Core_0
 *
 * Core_1 -> HW_Semaphore : Get_Semaphore
 * activate Core_1
 *
 * rnote over Core_1
 * Safe to operate on
 * shared resource
 * because of owning the
 * hardware semaphore
 * endrnote
 *
 * Core_1 -> HW_Semaphore : Release_Semaphore
 * deactivate Core_1
 * deactivate HW_Semaphore
 *
 * @enduml
 */
int vid_hwsema_getlock(__WORDADDRESS __local vid_hwsema_t *sema);

/**
 * @fn void vid_hwsema_rellock(__local vid_hwsema_t *sema)
 * @brief Release a lock previously acquired
 *
 * Return a lock implemented by a hardware semaphore that was previously acquired.
 *
 * @implements_hlc{LLLIB_HLC_03} @implements_su{LLLIB_SU_11}
 *
 * @param[in] sema Semaphore to use for lock release
 *
 * @startuml{vid_hwsema_rellock.png}
 * skinparam defaultTextAlignment center
 * start
 * :save compiler-used registers;
 * :get address of semaphore from semaphore id struct;
 * :wait for channel to become available;
 * :write semaphore to release it;
 * :restore compiler-used registers;
 * stop
 * @enduml
 *
 * ## Dynamic behaviour
 *
 * <center> ![Hardware Semaphore dynamic behavior](vid_hwsema_dynamic.png) </center>
 */
void vid_hwsema_rellock(__WORDADDRESS __local vid_hwsema_t *sema);

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
 *
 * @startuml{vid_get_core_id.png}
 * skinparam defaultTextAlignment center
 * start
 * :read core ID from DMEM;
 * :return core ID;
 * stop
 * @enduml
 */
uint vid_get_core_id(void);
#ifdef __cplusplus
}
#endif
#endif

//lint -restore
