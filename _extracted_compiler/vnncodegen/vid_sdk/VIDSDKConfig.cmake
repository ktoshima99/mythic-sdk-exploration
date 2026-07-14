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
# FILENAME:    VIDSDKConfig.cmake
#
# DESCRIPTION: videantis SDK cmake module config
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

# minimum required version for this module
set(_vidsdk_required_cmake_version 3.20.0)

# check if version requirement is met
if(${CMAKE_VERSION} VERSION_LESS ${_vidsdk_required_cmake_version})
  message(FATAL_ERROR "videantis SDK cmake module requires at least cmake version ${_vidsdk_required_cmake_version}")
endif()

# check if minimum required version of the project that uses the cmake module is less then the required version for the cmake module
# throw a warning for this case
if(${CMAKE_MINIMUM_REQUIRED_VERSION} VERSION_LESS ${_vidsdk_required_cmake_version})
  message(WARNING "Minimum required cmake version of project ${CMAKE_PROJECT_NAME} is less then required cmake version (${_vidsdk_required_cmake_version}) for the videantis SDK cmake module")
endif()

# set policy for relative paths of DEPFILE in add_custom_command()
# paths will be relative to CMAKE_CURRENT_BINARY_DIR
cmake_policy(SET CMP0116 NEW)

# variable containing the release version
set(_vidsdk_cmake_release_version 2026.2.0)
# variable containing the commit id
set(_vidsdk_cmake_commit_id 19232afe6b18a63ccfe00a8f1b450506d8da2ed1)

# include the videantis SDK module
include(${CMAKE_CURRENT_LIST_DIR}/cmake/modules/FindVIDSDK.cmake)
