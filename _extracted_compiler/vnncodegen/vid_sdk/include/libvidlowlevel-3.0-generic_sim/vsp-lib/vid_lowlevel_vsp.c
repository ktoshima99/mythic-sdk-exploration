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
 written authorization of videantis GmbH or an individual license agreementp
 with videantis GmbH is strictly forbidden.

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
*
* FILENAME: vid_lowlevel_vsp.c
*
* DESCRIPTION: lowlevel lib for v-SP
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#include "platform_define.h"
#include "vid_vsp_io.h"
#include "vid_lowlevelif.h"
#include "target_consts.h"

#define NULL 0

#define LOWLEVELIF_MAJOR_VERSION 2
#define LOWLEVELIF_MINOR_VERSION 3

#define MAX_MBP_ALLOC 30



/*
 * overlay memory related management data
 */

struct mbuf_status_t {
  unsigned configured;
  unsigned start_addr;
  unsigned bsize;
};

struct mbuf_status_t mbp_status = {0, 0, 0};
struct mbuf_status_t ocmbp_status = {0, 0, 0};

struct overlay_config_t {
        unsigned int offset;
        unsigned int xpos;
        unsigned int ypos;
        unsigned int width;
        unsigned int height;
        unsigned int format;
        unsigned int phys_addr;
        unsigned int size;
};


#define VID_MEM_MALLOC_ALIGN 8


#define DEB_LL(x...)  //printf(x)
#define DEB_PRINTF(x)

/*
 ******************************************************************************
 * API functions
 ******************************************************************************
 */


/**
 * \func vid_mem_init
 *
 * Return for the SW codec library the virtual overlay address memory SW codec
 * library calling function.
 *
 * \param addr: start address of memory segment
 *        size: size of memory segment
 *
 * \return VLL_OK: normal end
 *         VMP_E: error occurred
 *
 */
int vid_vsp_mem_init(unsigned int addr, unsigned int size) {
  if (!mbp_status.configured) {
      mbp_status.configured = 1;
      mbp_status.bsize = size;
      mbp_status.start_addr =  addr;
  }
    return VLL_OK;
}

/*
 * vid_ocmem_alloc
 *
 * Allocate memory segment from on-chip memory pool
 *
 * This function can be used once to allocate a segment
 * Subsequent calls after first successful call will have no effect
 *
 * \param addr: start address of memory segment
 *        size: size of memory segment
 *
 * \return VLL_OK: normal end
 *         VMP_E: error occurred
 *
 */
int vid_vsp_ocmem_init(unsigned int addr, unsigned int size) {

  if (!ocmbp_status.configured) {
     ocmbp_status.configured = 1;
     ocmbp_status.bsize = size;
     ocmbp_status.start_addr = addr;
  }
    return VLL_OK;
}

/*
 * ---------------------------------------------------------------------------
 * malloc/free system for overlay memory
 * ---------------------------------------------------------------------------
 */


/*
 * \func vid_mem_malloc
 *
 * \note     : Allocate memory segment from overlay memory pool
 *
 * \return
 *
 */
void* vid_mem_malloc(unsigned int size)
{
  unsigned returnptr ;


  if (size > mbp_status.bsize) {
    DEB_LL("ERROR vid_mem_malloc: failed to malloc memory.\n");
    return 0;
  }

  DEB_LL("vid_mem_malloc found start addr 0x%08x size %d\n",mbp_status.start_addr, mbp_status.bsize);

  returnptr = mbp_status.start_addr;
  mbp_status.bsize =  mbp_status.bsize - size;
  mbp_status.start_addr = mbp_status.start_addr + size;

  return (void*)returnptr;

}

void* vid_ocmem_malloc(unsigned int size)
{
  unsigned returnptr ;


  if (size > ocmbp_status.bsize) {
    DEB_LL("ERROR vid_ocmem_malloc: failed to malloc memory.\n");
    return 0;
  }

  DEB_LL("vid_ocmem_malloc found start addr 0x%08x size %d\n",ocmbp_status.start_addr, ocmbp_status.bsize);

  returnptr = ocmbp_status.start_addr;
  ocmbp_status.bsize =  ocmbp_status.bsize - size;
  ocmbp_status.start_addr = ocmbp_status.start_addr + size;

  return (void*)returnptr;

}

#ifndef _PLATFORM_ASR_CRANEGT_SIM
/* no hardware semaphores on ASR CraneGT */

/*
 * hardware semaphore lock request
 */
int32_t vid_hwsema_getlock(vid_hwsema_t *sema)
{
	int32_t retval;
	uint32_t val;

	if (NULL == sema) {
		return VLL_ERR;
	}

	if (sema->id > VLL_ID_HWSEMA_MAX) {
		return VLL_ERR;
	}

	val = dma_read(sema->addr);
	if (0 == val) {
	  retval = VLL_OK;
	}
	else {
	  retval = VLL_ERR;
	}

	return retval;
}

/*
 * hardware semaphore lock release
 */
int32_t vid_hwsema_rellock(vid_hwsema_t *sema)
{

	uint32_t val;

	if (NULL == sema) {
		return VLL_ERR;
	}

	if (sema->id > VLL_ID_HWSEMA_MAX) {
		return VLL_ERR;
	}

	val = 0;
	dma_write(sema->addr, val);

	return VLL_OK;
}
#endif
