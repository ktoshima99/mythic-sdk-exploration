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
# FILENAME:    VIDSDKConfig-version.cmake
#
# DESCRIPTION: videantis SDK cmake module config version
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

set(PACKAGE_VERSION 2026.2.0)

# check whether the requested PACKAGE_FIND_VERSION is compatible
if("${PACKAGE_VERSION}" VERSION_LESS "${PACKAGE_FIND_VERSION}")
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if ("${PACKAGE_VERSION}" VERSION_EQUAL "${PACKAGE_FIND_VERSION}")
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
