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

 *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *
 * FILENAME:            vid_dcif.h
 *
 * DESCRIPTION:         Prototypes for debug and control IF access
 *                      for Multi-Core Simulator
 *
 *+++++++++++++++++++++++++++++ FileHeaderEnd ++++++++++++++++++++++++++++++*/

#ifndef _VID_DCIF_H_
#define _VID_DCIF_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* API functions constants */
/// Return code OK
#define VID_DCIF_OK (0)
/// Return code ERROR
#define VID_DCIF_E  (-1)

/* host function prototypes */

int32_t vid_write_reg(uint32_t address, uint32_t value);
int32_t vid_read_reg(uint32_t address, uint32_t *value);

int32_t vid_set_signal(uint32_t signalid, uint32_t value);
int32_t vid_get_signal(uint32_t signalid, uint32_t *value);

/* init functions */

int32_t vid_dcif_init(void *ptr, uint32_t initflag);
void vid_dcif_quit();
void vid_semaphore_release(uint32_t deinitflag);

/* lib verbosity setting */

void vid_set_dcif_verbosity(int32_t level);

/* FIFO data types */
struct host2vidsim_read_reg_t {
  uint32_t address;
};

struct host2vidsim_write_reg_t {
  uint32_t address;
  uint32_t value;
};

struct host2vidsim_set_signal_t {
  uint32_t signalid;
  uint32_t value;
};

/// @brief Struct for host2vidsim fifo entry
struct host2vidsim_entry_t {
  uint32_t opid;
  union {
    struct host2vidsim_read_reg_t   read_op;
    struct host2vidsim_write_reg_t  write_op;
    struct host2vidsim_set_signal_t set_op;
  } op_entry;
};

/// @brief Struct for vidsim2host fifo entry
struct vidsim2host_entry_t {
  uint32_t address;
  uint32_t value;
};

/* internal defines */
#define VID_REG_READ_OP   (0x01u)
#define VID_REG_WRITE_OP  (0x02u)
#define VID_SIG_SET_OP    (0x03u)
#define VID_SIM_QUIT_OP   (0x10u)


/* direct FIFO access functions */

int32_t vid_host2vidsim_empty(void);
int32_t vid_host2vidsim_enqueue (struct host2vidsim_entry_t entry);
int32_t vid_host2vidsim_dequeue (struct host2vidsim_entry_t *entry);

int32_t vid_vidsim2host_enqueue (struct vidsim2host_entry_t entry);
int32_t vid_vidsim2host_dequeue (struct vidsim2host_entry_t *entry);

/* signal ID definitions */
#define VID_IRQOUT_SIG_OFFSET       (0x00u)
#define VID_IRQOUT_SIG_MP_MIN       (VID_IRQOUT_SIG_OFFSET + 0x00u)
#define VID_IRQOUT_SIG_MP_0         (VID_IRQOUT_SIG_OFFSET + 0x00u)
#define VID_IRQOUT_SIG_MP_1         (VID_IRQOUT_SIG_OFFSET + 0x01u)
#define VID_IRQOUT_SIG_MP_2         (VID_IRQOUT_SIG_OFFSET + 0x02u)
#define VID_IRQOUT_SIG_MP_3         (VID_IRQOUT_SIG_OFFSET + 0x03u)
#define VID_IRQOUT_SIG_MP_4         (VID_IRQOUT_SIG_OFFSET + 0x04u)
#define VID_IRQOUT_SIG_MP_5         (VID_IRQOUT_SIG_OFFSET + 0x05u)
#define VID_IRQOUT_SIG_MP_6         (VID_IRQOUT_SIG_OFFSET + 0x06u)
#define VID_IRQOUT_SIG_MP_7         (VID_IRQOUT_SIG_OFFSET + 0x07u)
#define VID_IRQOUT_SIG_MP_MAX       (VID_IRQOUT_SIG_OFFSET + 0x07u)
#define VID_IRQOUT_SIG_SP_MIN       (VID_IRQOUT_SIG_OFFSET + 0x14u)
#define VID_IRQOUT_SIG_SP_0         (VID_IRQOUT_SIG_OFFSET + 0x14u)
#define VID_IRQOUT_SIG_SP_1         (VID_IRQOUT_SIG_OFFSET + 0x15u)
#define VID_IRQOUT_SIG_SP_MAX       (VID_IRQOUT_SIG_OFFSET + 0x15u)
//note: vidsim refers to VID_IRQOUT_SIG_xx_0+core_index()

#define VID_IRQIN_SIG_OFFSET        (0x20u)
#define VID_IRQIN_SIG_MP_MIN        (VID_IRQIN_SIG_OFFSET + 0x00u)
#define VID_IRQIN_SIG_MP_0          (VID_IRQIN_SIG_OFFSET + 0x00u)
#define VID_IRQIN_SIG_MP_1          (VID_IRQIN_SIG_OFFSET + 0x01u)
#define VID_IRQIN_SIG_MP_2          (VID_IRQIN_SIG_OFFSET + 0x02u)
#define VID_IRQIN_SIG_MP_3          (VID_IRQIN_SIG_OFFSET + 0x03u)
#define VID_IRQIN_SIG_MP_4          (VID_IRQIN_SIG_OFFSET + 0x04u)
#define VID_IRQIN_SIG_MP_5          (VID_IRQIN_SIG_OFFSET + 0x05u)
#define VID_IRQIN_SIG_MP_6          (VID_IRQIN_SIG_OFFSET + 0x06u)
#define VID_IRQIN_SIG_MP_7          (VID_IRQIN_SIG_OFFSET + 0x07u)
#define VID_IRQIN_SIG_MP_MAX        (VID_IRQIN_SIG_OFFSET + 0x07u)
#define VID_IRQIN_SIG_SP_MIN        (VID_IRQIN_SIG_OFFSET + 0x14u)
#define VID_IRQIN_SIG_SP_0          (VID_IRQIN_SIG_OFFSET + 0x14u)
#define VID_IRQIN_SIG_SP_1          (VID_IRQIN_SIG_OFFSET + 0x15u)
#define VID_IRQIN_SIG_SP_MAX        (VID_IRQIN_SIG_OFFSET + 0x15u)
//note: vidsim refers to VID_IRQIN_SIG_xx_0+core_index()

/* DCIF shared memory segment layout definitions */

#define DCIF_SHARED_MEM_SIZE         (3u * 4096u)
#define DCIF_HOST2VIDSIM_FIFO_OFFSET (0u * 4096u)
#define DCIF_VIDSIM2HOST_FIFO_OFFSET (1u * 4096u)
#define DCIF_SIGNALS_OFFSET          (2u * 4096u)

/* internal, but externally useful data structures */

struct vidsim_signals_t {
  uint32_t mp_irqout[20u];
  uint32_t sp_irqout[8u];
};

/* disable semaphore depending on platform */
#ifdef _WIN32
#define _VID_DCIF_NO_SEMAPHORE
#endif

#ifdef __cplusplus
}
#endif

#endif // defined(_DCIF_H_)
