# ++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++
#
# CONFIDENTIAL AND PROPRIETARY INFORMATION
# Copyright 2004 - 2024 videantis GmbH
# All Rights Reserved
#
# This document contains confidential and proprietary information of videantis
# GmbH and is protected by copyright, trade secret and other local, state,
# federal, and international laws. Its receipt or possession does not convey
# any rights to reproduce, transfer, disclose or publish its contents, or to
# manufacture, commercially or non-commercially use or sell anything it may
# describe or contain. Reproduction, disclosure or any use without specific
# written authorization of videantis GmbH or an individual license agreement
# with videantis GmbH is strictly forbidden.
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# FILENAME:    libvidlowlevel-3.0-kiflex.cmake
#
# DESCRIPTION: Cmake file with definitions for libvidlowlevel-3.0-kiflex
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

# set variable with lowlevel library name
set(_lllib libvidlowlevel-3.0-kiflex)

# write commit id as cached variable
set(VIDSDK_LIBVIDLOWLEVEL-3.0-KIFLEX_COMMIT_ID 2026.2.0-21-g7d301f5d CACHE STRING "Commit id for videantis lowlevel library libvidlowlevel-3.0-kiflex")
# write description as cached variable
set(VIDSDK_LIBVIDLOWLEVEL-3.0-KIFLEX "videantis Low Level Library for KIFLEX" CACHE STRING "videantis lowlevel library libvidlowlevel-3.0-kiflex description")

# lowlevel library v-MP library dir
set(VID_LIBVIDLOWLEVEL-3.0-KIFLEX_VMP_LIB_DIR ${VIDSDK_LOWLEVELLIBRARY_PATH}/include/libvidlowlevel-3.0-kiflex/vmp-lib)
# lowlevel library include dir
set(VID_LIBVIDLOWLEVEL-3.0-KIFLEX_HOST_INCLUDE_DIRS ${VIDSDK_LOWLEVELLIBRARY_PATH}/include/libvidlowlevel-3.0-kiflex)
# lowlevel library library dir
set(VID_LIBVIDLOWLEVEL-3.0-KIFLEX_HOST_LIB_DIR ${VIDSDK_LOWLEVELLIBRARY_PATH}/lib)
# libraries to link for lowlevel library
set(VID_LIBVIDLOWLEVEL-3.0-KIFLEX_HOST_LIBS vidlowlevel-3.0-kiflex)
# lowlevel library host architecture
set(VID_LIBVIDLOWLEVEL-3.0-KIFLEX_HOST_ARCHITECTURE aarch64)
