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
# FILENAME:    vsp_tools.cmake
#
# DESCRIPTION: Support file for videantis SDK for v-SP tools
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

# v-SP tool chain is not mandatory
# looking for vspgcc executable
message(STATUS "Looking for vspgcc")
find_program(VSPGCC_EXECUTABLE vspgcc DOC "Find vspgcc")
if(VSPGCC_EXECUTABLE STREQUAL "VSPGCC_EXECUTABLE-NOTFOUND")
  message(STATUS "Looking for vspgcc - NOT FOUND")
else()
  message(STATUS "Looking for vspgcc - ${VSPGCC_EXECUTABLE}")

  # get vspgcc version
  execute_process(COMMAND ${VSPGCC_EXECUTABLE} --version OUTPUT_VARIABLE VSPGCC_VERSION ERROR_QUIET)

  # set a vspgcc version
  string(REGEX MATCH "Version:[ ]*([a-zA-Z0-9_\\.]+)" _ ${VSPGCC_VERSION})
  set(VSPGCC_VERSION ${CMAKE_MATCH_1} CACHE STRING "vspgcc version")
endif()

# looking for vspasm executable
message(STATUS "Looking for vspasm")
find_program(VSPASM_EXECUTABLE vspasm DOC "Find vspasm")
if(VSPASM_EXECUTABLE STREQUAL "VSPASM_EXECUTABLE-NOTFOUND")
  message(STATUS "Looking for vspasm - NOT FOUND")
else()
  message(STATUS "Looking for vspasm - ${VSPASM_EXECUTABLE}")

  # get vspasm version
  execute_process(COMMAND ${VSPASM_EXECUTABLE} --version OUTPUT_VARIABLE VSPASM_VERSION ERROR_QUIET)

  # set a vspasm version
  string(REGEX MATCH "Version:[ ]*([a-zA-Z0-9_\\.]+)" _ ${VSPASM_VERSION})
  set(VSPASM_VERSION ${CMAKE_MATCH_1} CACHE STRING "vspasm version")
endif()

# check if a v-SP tool is not found and set a variable to save this information
if((VSPGCC_EXECUTABLE STREQUAL "VSPGCC_EXECUTABLE-NOTFOUND") OR (VSPASM_EXECUTABLE STREQUAL "VSPASM_EXECUTABLE-NOTFOUND"))
  set(VIDSDK_HAS_VSP_TOOLCHAIN FALSE CACHE BOOL "videantis SDK v-SP tool chain included")
else()
  set(VIDSDK_HAS_VSP_TOOLCHAIN TRUE CACHE BOOL "videantis SDK v-SP tool chain included")
endif()

# check if v-SP tool chain was found
if(${VIDSDK_HAS_VSP_TOOLCHAIN})
  # set supported file types for vspgcc
  set(VSPGCC_SUPPORTED_FILE_TYPES C H CACHE STRING "vspgcc compiler supported file types")
  # set supported file types for vspasm
  set(VSPASM_SUPPORTED_FILE_TYPES ASM CACHE STRING "vspasm assembler supported file types")
endif()
