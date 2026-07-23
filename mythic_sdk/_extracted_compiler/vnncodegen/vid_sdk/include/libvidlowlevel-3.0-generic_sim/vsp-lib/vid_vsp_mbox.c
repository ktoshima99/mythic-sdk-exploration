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
* FILENAME: vid_vsp_mbox.c
*
* DESCRIPTION: mbox lowlevel lib implementation for v-SP
*
*++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++*/

#include "target_consts.h"
#include "target_lowlevel_consts.h"
#include "vid_vsp_ioregs.h"
#include "vid_lowlevelif.h"
#include "vid_vsp_io.h"

//#define VLL_DEBUG
//#define VSP_MBOX_DEBUG

/*
 * local helpers
 */
/* check if core id is valid */
#if VLL_ID_MP_MIN > 0
#define VLL_IS_VALID_CORE_ID(core_id)  ((((core_id) >= VLL_ID_MP_MIN)&&((core_id) <= VLL_ID_MP_MAX))||(((core_id) >= VLL_ID_SP_MIN)&&((core_id) <= VLL_ID_SP_MAX)) ? 1 : 0)
#else
#define VLL_IS_VALID_CORE_ID(core_id)  (((core_id) <= VLL_ID_MP_MAX)||(((core_id) >= VLL_ID_SP_MIN)&&((core_id) <= VLL_ID_SP_MAX)) ? 1 : 0)
#endif
/* map core id to range 0 ... VLL_NUM_CORES-1 */
#define VLL_CONDENSED_CORE_ID(core_id) (((core_id) < VLL_ID_SP_MIN) ? (core_id) : (VLL_NUM_MP + (core_id) - VLL_ID_SP_MIN))
#define VLL_CONDENSED_CORE_MP_ID(core_id) ((core_id))

#define to_be(val) (val)
#define to_le(val) (val)

#ifdef VLL_DEBUG
static int vid_vll_host_verbose = 15;
#define DEB_EE(x...) if (0 != (vid_vll_host_verbose & 0x01)) printf(x)          // enter and exit of functions
#define DEB_S(x...)  if (0 != (vid_vll_host_verbose & 0x02)) printf(x)          // simple debug messages
#define DEB_D(x...)  if (0 != (vid_vll_host_verbose & 0x04)) printf(x)          // detailed debug messages
#define DEB_E(x...)  if (0 != (vid_vll_host_verbose & 0x08)) printf(x)          // error messages
#else
#define DEB_EE(x...)
#define DEB_S(x...)
#define DEB_D(x...)
#define DEB_E(x...)
#endif


#define VLL_WAITTIME           100
#define VLL_MSG_VALID_MASK     1
#define VLL_MSG_MSGTYPE_VALID  0

#define VLL_MSG_MAX_IDX             (VLL_MSG_PER_BOX - 1)
#define VLL_MBOX_IDX_MASK           (VLL_MBOX_IDX_MOD - 1)

/*
 * default GPDATA register usage:
 * GPDATA0: Base mbox address for this core, used until init function called
 * GPDATA1: Core ID, used until init function called
 * GPDATA2: # of message sent by host
 * GPDATA3: # of messages received by v-SP core
 * GPDATA4: # of messages sent by v-SP core
 * GPDATA5: # of messages received by host
 */
volatile uint32_t *vid_vsp_irqout  = (volatile uint32_t *) VSP_ADDR_DCI_IRQOUT;
volatile uint32_t *vid_vsp_gpdata0 = (volatile uint32_t *) VSP_ADDR_GPDATA0;
volatile uint32_t *vid_vsp_gpdata1 = (volatile uint32_t *) VSP_ADDR_GPDATA1;
volatile uint32_t *vid_vsp_gpdata2 = (volatile uint32_t *) VSP_ADDR_GPDATA2;
volatile uint32_t *vid_vsp_gpdata3 = (volatile uint32_t *) VSP_ADDR_GPDATA3;
volatile uint32_t *vid_vsp_gpdata4 = (volatile uint32_t *) VSP_ADDR_GPDATA4;
volatile uint32_t *vid_vsp_gpdata5 = (volatile uint32_t *) VSP_ADDR_GPDATA5;
volatile uint32_t *vid_vsp_gpdata6 = (volatile uint32_t *) VSP_ADDR_GPDATA6;
volatile uint32_t *vid_vsp_gpdata7 = (volatile uint32_t *) VSP_ADDR_GPDATA7;
volatile uint32_t *vid_vsp_gpdata8 = (volatile uint32_t *) VSP_ADDR_GPDATA8;
volatile uint32_t *vid_vsp_gpdata9 = (volatile uint32_t *) VSP_ADDR_GPDATA9;

/* local message data types */
struct msg_t {
	vid_payload_t payload;
	uint32_t message_type;
	uint32_t valid_flag;
};

struct mbox_config_t {
	uint32_t VLL_MSG_EXT_BASEADDR_SND;
	uint32_t VLL_MSG_EXT_BASEADDR_RCV;
	uint32_t VLL_MSG_SND_COUNTER;
};

struct mcast_t {
	struct msg_t mbox[VLL_NUM_MP];
	uint64_t ready[VLL_NUM_MP];
};

/* shadow copy of the message boxes */
struct mbox_pair {
	uint32_t init_flag;
	uint32_t core_mbox_base_addr;
	uint32_t core_mbox_id_addr;
	uint32_t h2c_mbox_base;
	uint32_t h2c_posted_addr;
	uint32_t h2c_posted_copy;
	uint32_t h2c_consumed_addr;
	uint32_t c2h_mbox_base;
	uint32_t c2h_posted_addr;
	uint32_t c2h_consumed_addr;
	uint32_t c2h_consumed_copy;
};

/* static variables */
static struct msg_t mbox_shadow __attribute__ ((aligned (32)));
static struct mbox_config_t mbox;
static struct mbox_pair mboxes[VLL_NUM_CORES];


static unsigned int vid_mbox_whoami = VLL_ID_SP_0; /* default value */
volatile uint32_t g_sleepcounter = 0;
static uint32_t send_seq_number = 0;
#define SEND_SEQ_NUMBER_MAX 0x0f

int32_t vid_mbox_error = 0;

/* Mailbox base pointer */
static uint32_t mailbox_base = VLL_MBOX_BASE_DEFAULT;


/* ========================================================================== *
 * message API implementation                                                 *
 * ========================================================================== */

/*
 * vid_mbox_base()
 *
 * Set base address of mailbox memory segment
 *
 * returns: MBOX_ERR if known illegal address
 *          MBOX_OK otherwise
 */
int32_t vid_mbox_base(uint32_t address)
{
	int32_t rc;

	if (((address >= VLL_OCSRAM_START) &&
		 (address < (VLL_OCSRAM_START + VLL_OCSRAM_SIZE)) )
			||
		( (address >= VLL_SDRAM_START) &&
		  (address < (VLL_SDRAM_START + VLL_SDRAM_SIZE))) ) {
		mailbox_base = address;
		rc = VLL_MBOX_OK;
	}
	else {
		rc = VLL_MBOX_ERR;
	}
	return rc;
}
#if 0
/*
 * vid_mbox_init_vmp()
 *
 * initialize mailbox system to communicate from v-SP to v-MP
 * additional argument gived base address of mailbox in external memory,
 * needs to be allocated by v-SP before (size: 2*VLL_MBOX_SIZE)
 *
 * This function must be called for a core AFTER it was booted!
 *
 * returns: VLL_MBOX_ERR if illegal core id or other error
 *          VLL_MBOX_OK otherwise
 */
int32_t vid_mbox_init_vmp(uint32_t core_id, uint32_t local_mailbox_base) {
	DEB_EE("### vid_mbox_init_vmp (start) core id %d\n", core_id);

	struct mbox_pair *mbox_ptr;
    uint32_t m_core_id;

#if VLL_ID_MP_MIN > 0
    if (! ((core_id >= VLL_ID_MP_MIN) && (core_id <= VLL_ID_MP_MAX))) {
#else
	if (core_id > VLL_ID_MP_MAX) {
#endif
    	/* we must be called only for v-MPs ! */
		return VLL_MBOX_ERR;
	}

    m_core_id = VLL_CONDENSED_CORE_MP_ID(core_id);

    mbox_ptr = &(mboxes[m_core_id]);

    if (1 == mbox_ptr->init_flag) {
  		DEB_E("--- Error vid_mem_init: Re-initialization detected!\n");
        DEB_EE("### vid_mbox_init_vmp (end)\n");
        return VLL_MBOX_ERR;
    }

    /* initalize shadow mailbox */
    mbox_ptr->core_mbox_base_addr = VMP_CTRL_BASE + core_id * VMP_REGWIN_SIZE + GPDATA0_OFFSET;
    mbox_ptr->core_mbox_id_addr = VMP_CTRL_BASE + core_id * VMP_REGWIN_SIZE + GPDATA1_OFFSET;
    mbox_ptr->h2c_mbox_base = local_mailbox_base + VLL_MBOX_SIZE;//mailbox_base + mbox_out_bases[m_core_id];
    mbox_ptr->h2c_posted_addr = VMP_CTRL_BASE + core_id * VMP_REGWIN_SIZE + GPDATA2_OFFSET;
    mbox_ptr->h2c_posted_copy = 0;
    mbox_ptr->h2c_consumed_addr = VMP_CTRL_BASE + core_id * VMP_REGWIN_SIZE + GPDATA3_OFFSET;
    mbox_ptr->c2h_mbox_base = local_mailbox_base; //mailbox_base + mbox_in_bases[m_core_id];
    mbox_ptr->c2h_posted_addr = VMP_CTRL_BASE + core_id * VMP_REGWIN_SIZE + GPDATA4_OFFSET;
    mbox_ptr->c2h_consumed_addr = VMP_CTRL_BASE + core_id * VMP_REGWIN_SIZE + GPDATA5_OFFSET;
    mbox_ptr->c2h_consumed_copy = 0;

    DEB_D("Send config data to core %u - mbox base %08x ID %u\n", core_id, mbox_ptr->c2h_mbox_base, core_id);
    /* The other v-MP cores must have been booted at this point ! */
    dma_write(mbox_ptr->core_mbox_id_addr, core_id | 0x80000000 ); /* set MSB in ID to prevent v-MP from asserting IRQOUT on send */
    dma_write(mbox_ptr->core_mbox_base_addr, mbox_ptr->c2h_mbox_base);

    mbox_ptr->init_flag = 1;

	DEB_EE("### vid_mbox_init_vmp (end)\n");
	return VLL_MBOX_OK;
}
#endif

/*
 * vid_mbox_init()
 *
 * initialize mailbox system
 *
 *
 * returns: VLL_MBOX_ERR if illegal core id or other error
 *          VLL_MBOX_OK otherwise
 */
int32_t vid_mbox_init(uint32_t core_id) {
	DEB_EE("### %d (start) core id %d\n", __func__, core_id);

	if (core_id != VLL_ID_HOST) {
		return VLL_MBOX_ERR;
	}

	/* communicate with Host */

	/*
	 * The core gets the following data passed from the core that
	 * initializes the mailboxes:
	 * GPDATA0: Base address of mailbox pair to use
	 * GPDATA1: Core ID
	 *
	 * The core triggers for GPDATA0 != 0, and then reads GPDATA1
	 */
	do {
		mbox.VLL_MSG_EXT_BASEADDR_SND = *vid_vsp_gpdata0;
	} while (mbox.VLL_MSG_EXT_BASEADDR_SND ==0);
	mbox.VLL_MSG_EXT_BASEADDR_RCV = mbox.VLL_MSG_EXT_BASEADDR_SND + VLL_MBOX_SIZE;

	/* save core ID passed in GPDATA1 */
	vid_mbox_whoami = *vid_vsp_gpdata1;

	/* initialize send message counter */
	mbox.VLL_MSG_SND_COUNTER = 0;

	DEB_EE("### %d (cfg) %08x %08x %d\n", __func__, mbox.VLL_MSG_EXT_BASEADDR_SND, mbox.VLL_MSG_EXT_BASEADDR_RCV, vid_mbox_whoami);

	DEB_EE("### %d (end)\n", __func__);
	return VLL_MBOX_OK;
}

/*
 * vid_mbox_send()
 *
 * send message using mailbox system
 *
 *
 * returns: VLL_MBOX_WOULDBLOCK if function would block and non-blocking is selected
 *          VLL_MBOX_ERR        if general error
 *          VLL_MBOX_OK         otherwise
 */
int32_t vid_mbox_send( uint32_t core_id, uint32_t msg_type, vid_payload_t *payload, uint32_t blocking_flag) {

	DEB_EE("### %d (start) core id %d\n", __func__, core_id);

	volatile unsigned int *ptr_dci_irqen = (volatile unsigned int *) VSP_ADDR_DCI_IRQEN;
	volatile unsigned int *ptr_dci_irqstatus = (volatile unsigned int *) VSP_ADDR_DCI_IRQSTATUS;

	unsigned int i;

	unsigned int msg_ext_addr;
	int32_t msg_sent, msg_rcvd, msg_diff;

	if (VLL_ID_HOST == core_id) {
		/* communicate with Host */
		/*
		 * GPDATA4: # of messages sent by v-SP core
		 * GPDATA5: # of messages received by host
		 *
		 */

		msg_sent = *vid_vsp_gpdata4;
		/* prepare trigger on GPDATA5 - messages received by host */
		*ptr_dci_irqen = (*ptr_dci_irqen) | (1 << 5);
		msg_rcvd = *vid_vsp_gpdata5;
		msg_diff = msg_sent - msg_rcvd;
		if (msg_diff < 0) {
			msg_diff += VLL_MBOX_IDX_MOD;
		}
		if ((VLL_MBOX_NONBLOCKING == blocking_flag) && (VLL_MSG_PER_BOX == msg_diff)) {
			/* not empty slots at host, call would block */
			/* remove trigger on GPDATA5 */
	  		*ptr_dci_irqen = (*ptr_dci_irqen) ^ (1 << 5);
	  		/* clear GPDATA5 DCI IRQ Source */
	  		*ptr_dci_irqstatus = ~ (1 << 5);
			return VLL_MBOX_WOULDBLOCK;
		}
		if (VLL_MSG_PER_BOX == msg_diff) {

#ifdef VSP_MBOX_DEBUG
			printf("VSP: Waiting for DCI send IRQ!\n");
#endif
			/* wait for DCI0 IRQ */
			vid_wait_irq_dci0 ;

			msg_rcvd = *vid_vsp_gpdata5;
			msg_diff = msg_sent - msg_rcvd;
			if (msg_diff < 0) {
				msg_diff += VLL_MBOX_IDX_MOD;
			}
		}
		/* remove trigger on GPDATA5 */
  		*ptr_dci_irqen = (*ptr_dci_irqen) ^ (1 << 5);
  		/* clear GPDATA5 DCI IRQ Source */
  		*ptr_dci_irqstatus = ~ (1 << 5);

#ifdef VSP_MBOX_DEBUG
		if (VLL_MSG_PER_BOX == msg_diff) {
			printf("VSP: FATAL wakeup in vid_mbox_send!\n");
		}
#endif
		/* There is at least one free slow available, calculate address */
		msg_ext_addr = mbox.VLL_MSG_EXT_BASEADDR_SND + (VLL_MSG_MAX_SIZE * (msg_sent & VLL_MSG_MAX_IDX));

		/* copy payload to shadow mailbox */
		for (i=0; i<VLL_MSG_PAYLOAD_SIZE/8; i++) {
			mbox_shadow.payload.pl_ui32[i].s1 = payload->pl_ui32[i].s1;
			mbox_shadow.payload.pl_ui32[i].s0 = payload->pl_ui32[i].s0;
		}

		/* and set valid bit and message type */
		mbox_shadow.message_type = msg_type;
		mbox_shadow.valid_flag = VLL_MSG_VALID_MASK;


		/* send message including valid flag and message type */
		dma_write_block(msg_ext_addr, (char *) &mbox_shadow, VLL_MSG_MAX_SIZE);

		/* we have sent a new message , increment counter */
		/* sequence counter values are 0..15 */
		mbox.VLL_MSG_SND_COUNTER++;
		mbox.VLL_MSG_SND_COUNTER = mbox.VLL_MSG_SND_COUNTER & VLL_MBOX_IDX_MASK;

		dma_wait();

		/* adjust posted messages counter */
		msg_sent = *vid_vsp_gpdata4;
		msg_sent++;
		msg_sent = msg_sent & VLL_MBOX_IDX_MASK;
		*vid_vsp_gpdata4 = msg_sent;

		/* re-read message since we have posted writes */
		/* after read is finished, message is in memory for sure */
		dma_read_block(msg_ext_addr, (char *) &mbox_shadow, 4);
		dma_wait();

		/* assert IRQOUT, if MSB of whoami is not set */
		if ((vid_mbox_whoami & 0x80000000) == 0) {
			*vid_vsp_irqout = 0x01;
		}
	} else {
		/* check ID */
#if VLL_ID_MP_MIN > 0
		if ((core_id < VLL_ID_MP_MIN) || (core_id > VLL_ID_MP_MAX)) {
#else
		if (core_id > VLL_ID_MP_MAX) {
#endif
			return VLL_MBOX_ERR;
		}
		/* communicate with other v-MP */
		uint32_t loopflag;
		struct mbox_pair *mbox_ptr;
		uint32_t ptr_v;
		uint32_t m_core_id;

		uint32_t h2c_consumed;
		uint32_t h2c_posted_prime;

		m_core_id = VLL_CONDENSED_CORE_MP_ID(core_id);
		mbox_ptr = &(mboxes[m_core_id]);

		/* check if properly initalized */
		if (1 != mbox_ptr->init_flag) {
			DEB_E("--- Error vid_mbox_send: Mbox not initialized!\n");
			DEB_EE("### %d (end)\n", __func__);
			return VLL_MBOX_ERR;
		}

		loopflag = 1;
		while(1 == loopflag) {
			h2c_consumed = dma_read(mbox_ptr->h2c_consumed_addr);

			if (mbox_ptr->h2c_posted_copy < h2c_consumed) {
				h2c_posted_prime = mbox_ptr->h2c_posted_copy + VLL_MBOX_IDX_MOD;
			}
			else {
				h2c_posted_prime = mbox_ptr->h2c_posted_copy;
			}

			if ((h2c_posted_prime - h2c_consumed) < VLL_MSG_PER_BOX) {
				/* Space available in mailbox */
				mbox_shadow.payload = *payload;
				mbox_shadow.message_type = to_be(msg_type);
				mbox_shadow.valid_flag = to_be((send_seq_number << 8) | 0x01);
				send_seq_number = (send_seq_number + 1) & SEND_SEQ_NUMBER_MAX;
				ptr_v = mbox_ptr->h2c_mbox_base + (mbox_ptr->h2c_posted_copy % VLL_MSG_PER_BOX) * sizeof(struct msg_t);
				dma_write_block(ptr_v, (char *) &mbox_shadow, sizeof(struct msg_t));

				mbox_ptr->h2c_posted_copy = (mbox_ptr->h2c_posted_copy + 1) % VLL_MBOX_IDX_MOD;
				dma_write(mbox_ptr->h2c_posted_addr, mbox_ptr->h2c_posted_copy);

				loopflag = 0;
			}
			else {
				/* No space currently available in mailbox */
				if (VLL_MBOX_NONBLOCKING == blocking_flag) {
					DEB_EE("### %d (end) blocking_flag==0\n", __func__);
					return VLL_MBOX_WOULDBLOCK;
				}
				else {
					// Continue checking for free slots at v-MP
				}
			}
		}
	}

	DEB_EE("### %d (end)\n", __func__);

	return VLL_MBOX_OK;
}

/*
 * vid_mbox_rcv()
 *
 * receive message using mailbox system
 *
 * after successful receive, the incoming message is still in the box.
 * It needs to be released after processing (if necessary) with vid_mbox_rel()
 *
 * returns: MBOX_WOULDBLOCK if function would block and non-blocking is selected
 *          MBOX_ERR        if general error
 *          MBOX_OK         otherwise, i.e., successful receive
 */
int32_t vid_mbox_rcv(uint32_t core_id, uint32_t *msg_type, uint32_t *handle, vid_payload_t *payload, uint32_t blocking_flag) {
	DEB_EE("### %d (start) core id %d\n", __func__, core_id);


		volatile unsigned int *ptr_dci_irqen = (volatile unsigned int *) VSP_ADDR_DCI_IRQEN;
		volatile unsigned int *ptr_dci_irqstatus = (volatile unsigned int *) VSP_ADDR_DCI_IRQSTATUS;

	unsigned int i;


	unsigned int msg_ext_addr;
	int32_t msg_sent, msg_rcvd;
	// printf("VSP: vid_mbox_rcv enter\n");
	if (VLL_ID_HOST == core_id) {
		/* communicate with Host */
		/*
		 * GPDATA2: # of message sent by host
		 * GPDATA3: # of messages received by v-SP core
		 */
		msg_rcvd = *vid_vsp_gpdata3;

		*ptr_dci_irqen = (*ptr_dci_irqen) | (1 << 2); /* prepare trigger on GPDATA2 - messages sent by host */
		msg_sent = *vid_vsp_gpdata2;
		if ((VLL_MBOX_NONBLOCKING == blocking_flag) && (msg_sent == msg_rcvd)) {
		        /* not message from host, call would block */
				/* remove trigger on GPDATA2 */
		        *ptr_dci_irqen = (*ptr_dci_irqen) ^ (1 << 2);
		        /* clear GPDATA2 DCI IRQ Source */
		        *ptr_dci_irqstatus = ~ (1 << 2);
		        return VLL_MBOX_WOULDBLOCK;
		}

		if (msg_sent == msg_rcvd) {
#ifdef VSP_MBOX_DEBUG
			printf("VSP: Waiting for DCI rcv IRQ!\n");
#endif
			/* wait for DCI0 IRQ */
			vid_wait_irq_dci0 ;
			msg_sent = *vid_vsp_gpdata2;
		}
		/* remove trigger on GPDATA2 */
		*ptr_dci_irqen = (*ptr_dci_irqen) ^ (1 << 2);
		/* clear GPDATA2 DCI IRQ Source */
		*ptr_dci_irqstatus = ~ (1 << 2);


#ifdef VSP_MBOX_DEBUG
		if (msg_sent == msg_rcvd) {
			printf("VSP: FATAL wakeup in vid_mbox_rcv!\n");
		}
#endif
		/* There is at least one message available, calculate address */
		msg_ext_addr = mbox.VLL_MSG_EXT_BASEADDR_RCV + (VLL_MSG_MAX_SIZE * (msg_rcvd & VLL_MSG_MAX_IDX));

		/* load message to mbox_shadow */
		dma_read_block(msg_ext_addr, (char *) &mbox_shadow, VLL_MSG_MAX_SIZE);

		/* copy payload to shadow mailbox */
		for (i=0; i<VLL_MSG_PAYLOAD_SIZE/8; i++) {
			payload->pl_ui32[i].s1 = mbox_shadow.payload.pl_ui32[i].s1;
			payload->pl_ui32[i].s0 = mbox_shadow.payload.pl_ui32[i].s0;
		}

		/* message loaded, access message type at end of message */
		*msg_type = mbox_shadow.message_type;

	} else {
		/* check ID */
#if VLL_ID_MP_MIN > 0
		if ((core_id < VLL_ID_MP_MIN) || (core_id > VLL_ID_MP_MAX)) {
#else
		if (core_id > VLL_ID_MP_MAX) {
#endif
			return VLL_MBOX_ERR;
		}
		/* communicate with other v-MP */
		uint32_t loopflag;
		struct mbox_pair *mbox_ptr;
		uint32_t ptr_v;
		uint32_t m_core_id;

		uint32_t c2h_posted;
		uint32_t c2h_posted_prime;

		m_core_id = VLL_CONDENSED_CORE_MP_ID(core_id);
		mbox_ptr = &(mboxes[m_core_id]);

		/* check if properly initalized */
		if (1 != mbox_ptr->init_flag) {
			DEB_E("--- Error vid_mbox_rcv: Mbox not initialized!\n");
			DEB_EE("### %d (end)\n", __func__);
			return VLL_MBOX_ERR;
		}

		loopflag = 1;
		while(1 == loopflag) {
			c2h_posted = dma_read(mbox_ptr->c2h_posted_addr);

			if (c2h_posted < mbox_ptr->c2h_consumed_copy) {
				c2h_posted_prime = c2h_posted + VLL_MBOX_IDX_MOD;
			}
			else {
				c2h_posted_prime = c2h_posted;
			}

			if ((c2h_posted_prime - mbox_ptr->c2h_consumed_copy) > 0) {
				/* at least one message available in mailbox */
				ptr_v = mbox_ptr->c2h_mbox_base + (mbox_ptr->c2h_consumed_copy % VLL_MSG_PER_BOX) * sizeof(struct msg_t);
				dma_read_block(ptr_v, (char *) &mbox_shadow, sizeof(struct msg_t));
				*msg_type = to_le(mbox_shadow.message_type);
				*payload = mbox_shadow.payload;
				*handle = mbox_ptr->c2h_consumed_copy;
				loopflag = 0;
			}
			else {
				/* no messages available in mailbox */
				if (VLL_MBOX_NONBLOCKING == blocking_flag) {
					DEB_EE("### %d (end) blocking_flag==0\n", __func__);
					return VLL_MBOX_WOULDBLOCK;
				}
				else {
					// Continue checking for message available from v-MP
				}
			}
		}
	}

	DEB_EE("### %d (end)\n", __func__);
	return VLL_MBOX_OK;
}


/*
 * vid_mbox_rel()
 *
 * release a message from the incoming mailbox
 *
 *
 * returns: VLL_MBOX_ERR    if general error
 *          VLL_MBOX_OK     otherwise
 */
int32_t vid_mbox_rel(uint32_t core_id, uint32_t handle) {
	DEB_EE("### %d (start) core id %d\n", __func__, core_id);

	int32_t msg_rcvd;

	if (VLL_ID_HOST == core_id) {
		/* communicate with Host */
		/*
		 * GPDATA3: # of messages received by v-MP core
		 */
		msg_rcvd = *vid_vsp_gpdata3;
		msg_rcvd++;
		msg_rcvd = msg_rcvd & VLL_MBOX_IDX_MASK;
		*vid_vsp_gpdata3 = msg_rcvd;
	} else {

		DEB_EE("### %d (start) core id %d\n", __func__, core_id);
		struct mbox_pair *mbox_ptr;

		uint32_t m_core_id;

		/* check ID */
		if (! VLL_IS_VALID_CORE_ID(core_id)) {
			DEB_E("--- Error vid_mbox_rel: Illegal Core ID!\n");
			DEB_EE("### %d (end)\n", __func__);
			return VLL_MBOX_ERR;
		}
		m_core_id = VLL_CONDENSED_CORE_ID(core_id);
		mbox_ptr = &(mboxes[m_core_id]);

		/* check if properly initalized */
		if (1 != mbox_ptr->init_flag) {
			DEB_E("--- Error vid_mbox_rel: Mbox not initialized!\n");
			DEB_EE("### %d (end)\n", __func__);
			return VLL_MBOX_ERR;
		}

		/* check if valid handle presented */
		if (handle != mbox_ptr->c2h_consumed_copy) {
			DEB_E("--- Error vid_mbox_rel: Invalid handle!\n");
			DEB_EE("### %d (end)\n", __func__);
			return VLL_MBOX_ERR;
		}

		mbox_ptr->c2h_consumed_copy = (mbox_ptr->c2h_consumed_copy + 1) % VLL_MBOX_IDX_MOD;
		dma_write(mbox_ptr->c2h_consumed_addr, mbox_ptr->c2h_consumed_copy);
	}

	DEB_EE("### %d (end)\n", __func__);
	return VLL_MBOX_OK;
}

uint32_t vid_get_core_id(void)
{
  return ((vid_mbox_whoami << 1) >> 1);
}
