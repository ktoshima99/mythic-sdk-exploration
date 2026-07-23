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
# FILENAME:    FindVIDSDK.cmake
#
# DESCRIPTION: Module for integrating the videantis SDK into CMake
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

###
# Setup videantis SDK cmake module
###
if (NOT VIDSDK)
  message(STATUS "Looking for videantis SDK")
  if (DEFINED ENV{VID_SDK_ROOT})
    # save VID_SDK_ROOT as cmake variable for easier access
    # TODO: would be more consistent with the variable VIDSDK_ROOT,
    #       but this would change some cmake builtin behavior. Investigate later...
    set(VID_SDK_ROOT $ENV{VID_SDK_ROOT} CACHE PATH "Root directory of videantis SDK")
    # get folder of videantis SDK
    get_filename_component(_vidsdk_folder ${VID_SDK_ROOT} NAME)
    # set VIDSDK variable with folder that contains SDK
    set(VIDSDK ${_vidsdk_folder} CACHE STRING "videantis SDK")
    message(STATUS "Looking for videantis SDK - ${VID_SDK_ROOT}")

    # set VIDSDK_CMAKE_MODULE_VERSION with release version of cmake module
    set(VIDSDK_CMAKE_MODULE_VERSION ${_vidsdk_cmake_release_version} CACHE STRING "Version of videantis SDK cmake module")

    # set VIDSDK_CMAKE_MODULE_COMMIT_ID with commit id of cmake module
    set(VIDSDK_CMAKE_MODULE_COMMIT_ID ${_vidsdk_cmake_commit_id} CACHE STRING "Commit id of videantis SDK cmake module")

    # set VIDSDK_CMAKE_MODULE_DIR with the directory of all module files
    set(VIDSDK_CMAKE_MODULE_DIR ${CMAKE_CURRENT_LIST_DIR}/VIDSDK CACHE PATH "Directory of videantis SDK cmake module files")

    # looking for vmpcc executable
    message(STATUS "Looking for vmpcc")
    find_program(VMPCC_EXECUTABLE vmpcc DOC "Find vmpcc")
    if(VMPCC_EXECUTABLE STREQUAL "VMPCC_EXECUTABLE-NOTFOUND")
      message(FATAL_ERROR "vmpcc NOT FOUND")
    endif()
    message(STATUS "Looking for vmpcc - ${VMPCC_EXECUTABLE}")

    # get vmpcc version
    execute_process(COMMAND ${VMPCC_EXECUTABLE} --version OUTPUT_VARIABLE VMPCC_VERSION ERROR_QUIET)

    # set a LLVM version
    string(REGEX MATCH "[ ]*clang version[ ]*([0-9\\.]+)" _ ${VMPCC_VERSION})
    set(VMPCC_LLVM_VERSION ${CMAKE_MATCH_1} CACHE STRING "LLVM version used in vmpcc")
    message(STATUS "Looking for vmpcc - LLVM version ${VMPCC_LLVM_VERSION}")

    # set a LLVM major version
    string(REGEX MATCH "([0-9]+)\\.([0-9\\.]+)" _ ${VMPCC_LLVM_VERSION})
    set(VMPCC_LLVM_MAJOR ${CMAKE_MATCH_1} CACHE STRING "LLVM version major used in vmpcc")

    # set a vmpcc version
    string(REGEX MATCH "Version:[ ]*([a-zA-Z0-9_\\.]+)" _ ${VMPCC_VERSION})
    set(VMPCC_VERSION ${CMAKE_MATCH_1} CACHE STRING "vmpcc version")

    # set supported file types for vmpcc
    set(VMPCC_SUPPORTED_FILE_TYPES CL C CPP CACHE STRING "vmpcc compiler supported file types")

    # looking for vmpasm executable
    message(STATUS "Looking for vmpasm")
    find_program(VMPASM_EXECUTABLE vmpasm DOC "Find vmpasm")
    if(VMPASM_EXECUTABLE STREQUAL "VMPASM_EXECUTABLE-NOTFOUND")
      message(FATAL_ERROR "vmpasm NOT FOUND")
    endif()
    message(STATUS "Looking for vmpasm - ${VMPASM_EXECUTABLE}")

    # get vmpasm version
    execute_process(COMMAND ${VMPASM_EXECUTABLE} --version OUTPUT_VARIABLE VMPASM_VERSION ERROR_QUIET)

    # set a vmpasm version
    string(REGEX MATCH "Version:[ ]*([a-zA-Z0-9_\\.]+)" _ ${VMPASM_VERSION})
    set(VMPASM_VERSION ${CMAKE_MATCH_1} CACHE STRING "vmpasm version")

    # set supported file types for vmpasm
    set(VMPASM_SUPPORTED_FILE_TYPES ASM INC CACHE STRING "vmpasm assembler supported file types")

    # set path for cmake files to check for v-SP tools
    set(_tools_vsp_cmake ${VIDSDK_CMAKE_MODULE_DIR}/tools_vsp.cmake)
    # if the cmake file for v-SP tools exists (no installed, when v-SP tool chain excluded)
    if(EXISTS ${_tools_vsp_cmake})
      # include file to check for v-SP tools
      include(${_tools_vsp_cmake})
    else()
      # set cached variable to provided information that v-SP tool chain is not included
      set(VIDSDK_HAS_VSP_TOOLCHAIN FALSE CACHE BOOL "videantis SDK v-SP tool chain included")
    endif()

    # looking for bin2h executable
    message(STATUS "Looking for bin2h")
    find_program(BIN2H_EXECUTABLE bin2h DOC "Find bin2h")
    if(BIN2H_EXECUTABLE STREQUAL "BIN2H_EXECUTABLE-NOTFOUND")
      message(STATUS "Looking for bin2h - NOT FOUND")
    else()
      message(STATUS "Looking for bin2h - ${BIN2H_EXECUTABLE}")
    endif()

    # looking for vidsim executable
    message(STATUS "Looking for vidsim")
    find_program(VIDSIM_EXECUTABLE vidsim DOC "Find vidsim")
    if(VIDSIM_EXECUTABLE STREQUAL "VIDSIM_EXECUTABLE-NOTFOUND")
      message(FATAL_ERROR "vidsim NOT FOUND")
    endif()
    message(STATUS "Looking for vidsim - ${VIDSIM_EXECUTABLE}")

    # looking for vidsim-cct executable
    message(STATUS "Looking for vidsim-cct")
    find_program(VIDSIM-CCT_EXECUTABLE vidsim-cct DOC "Find vidsim-cct")
    if(VIDSIM-CCT_EXECUTABLE STREQUAL "VIDSIM-CCT_EXECUTABLE-NOTFOUND")
      message(FATAL_ERROR "vidsim-cct NOT FOUND")
    endif()
    message(STATUS "Looking for vidsim-cct - ${VIDSIM-CCT_EXECUTABLE}")

    # looking for vidsim-man executable
    message(STATUS "Looking for vidsim-man")
    find_program(VIDSIM-MAN_EXECUTABLE vidsim-man DOC "Find vidsim-man")
    if(VIDSIM-MAN_EXECUTABLE STREQUAL "VIDSIM-MAN_EXECUTABLE-NOTFOUND")
      message(FATAL_ERROR "vidsim-man NOT FOUND")
    endif()
    message(STATUS "Looking for vidsim-man - ${VIDSIM-MAN_EXECUTABLE}")

    # looking for vidsim-dbg executable
    message(STATUS "Looking for vidsim-dbg")
    find_program(VIDSIM-DBG_EXECUTABLE vidsim-dbg DOC "Find vidsim-dbg")
    if(VIDSIM-DBG_EXECUTABLE STREQUAL "VIDSIM-DBG_EXECUTABLE-NOTFOUND")
      message(FATAL_ERROR "vidsim-dbg NOT FOUND")
    endif()
    message(STATUS "Looking for vidsim-dbg - ${VIDSIM-DBG_EXECUTABLE}")

    # include file to test and define capabilities of videantis simulator
    include(${VIDSDK_CMAKE_MODULE_DIR}/vidsim.cmake)

    # set default path for lowlevel library to root folder of videantis SDK
    # the path can be overwritten, when defining VIDSDK_LOWLEVELLIBRARY_PATH as command line argument
    set(VIDSDK_LOWLEVELLIBRARY_PATH ${VID_SDK_ROOT} CACHE PATH "Path of videantis lowlevel library")

    # looking for available videantis lowlevel libraries within the SDK
    message(STATUS "Looking for videantis lowlevel library")
    message(STATUS "Looking for videantis lowlevel library - ${VIDSDK_LOWLEVELLIBRARY_PATH}")
    file(GLOB _cmake_files_lllibrary "${VIDSDK_LOWLEVELLIBRARY_PATH}/lib/cmake/*.cmake")
    foreach(_cmake_file_lllibrary ${_cmake_files_lllibrary})
      set(_lllib LLLIB-NOTFOUND)
      # include cmake file of lowlevel library
      include(${_cmake_file_lllibrary})
      # check if cmake lowlevel library file had required variable defined
      if(NOT ${_lllib} STREQUAL "LLLIB-NOTFOUND")
        # append lowlevel library to list of lowlevel libraries
        list(APPEND _vid_lllibrary ${_lllib})

        message(STATUS "Looking for videantis lowlevel library - ${_lllib}")
      endif()
    endforeach(_cmake_file_lllibrary)

    # check if a lowlevel library is found
    if(NOT DEFINED _vid_lllibrary)
      message(FATAL_ERROR "No videantis lowlevel library is found under ${VIDSDK_LOWLEVELLIBRARY_PATH}/lib/cmake!")
    endif()

    # write cached variable with the list of found lowlevel libraries
    set(VIDSDK_LOWLEVELLIBRARY ${_vid_lllibrary} CACHE STRING "videantis SDK lowlevel library")

    # set default videantis lowlevel library v-MP firmware size and number of v-MP cores
    # the values can be overwritten, when defining VIDSDK_VLL_MP_FW_SIZE or VIDSDK_VLL_NUM_MP as command line argument
    set(VIDSDK_VLL_MP_FW_SIZE "0x00020000" CACHE STRING "videantis lowlevel library v-MP firmware size")
    set(VIDSDK_VLL_NUM_MP "8" CACHE STRING "videantis lowlevel library number of v-MPs")

    # check if lowlevel library constants for v-SP components are required
    if(${VIDSDK_HAS_VSP_TOOLCHAIN})
      # set default videantis lowlevel library v-SP firmware size and number of v-SP cores
      # the values can be overwritten, when defining VIDSDK_VLL_SP_FW_SIZE or VIDSDK_VLL_NUM_SP as command line argument
      set(VIDSDK_VLL_SP_FW_SIZE "0x00020000" CACHE STRING "videantis lowlevel library v-SP firmware size")
      set(VIDSDK_VLL_NUM_SP "2" CACHE STRING "videantis lowlevel library number of v-SPs")
    endif()

    # list of variable to parse from target_consts.h of lowlevel library
    set(VIDSDK_VLL_TARGET_CONSTS VLL_SDRAM_START VLL_SDRAM_SIZE VLL_OCSRAM_START VLL_OCSRAM_SIZE VMP_CTRL_BASE CACHE STRING "variables to parse from target_consts.h of lowlevel library")
    # list of mappings for parsed variables from target_consts.h of lowlevel library
    set(VIDSDK_VLL_TARGET_CONSTS_MAP SDRAM_ADDR SDRAM_SIZE OCSRAM_ADDR OCSRAM_SIZE DEBUGIF_ADDR CACHE STRING "mappings for parsed variables from target_consts.h of lowlevel library")
  else()
    message(FATAL_ERROR "Environment variable VID_SDK_ROOT is not set, cannot find videantis SDK")
  endif()
else()
  # check if a videantis SDK is loaded
  if (DEFINED ENV{VID_SDK_ROOT})
    # check if loaded SDK matches cached SDK
    if(NOT ${VID_SDK_ROOT} STREQUAL $ENV{VID_SDK_ROOT})
      # get folder of videantis SDK
      get_filename_component(_vidsdk_folder $ENV{VID_SDK_ROOT} NAME)
      # raise error message
      message(FATAL_ERROR "The loaded videantis SDK (${_vidsdk_folder}) does not match with the SDK in the cmake cache (${VIDSDK}). "
        "Recreate cmake build directory to use the loaded videantis SDK or switch back to cached SDK!")
    endif()
  # no SDK is loaded
  else()
    message(FATAL_ERROR "No videantis SDK is loaded! To continue load videantis SDK: ${VIDSDK}")
  endif()

  # load all known lowlevel libraries
  foreach(_lllib ${VIDSDK_LOWLEVELLIBRARY})
    # expected path to cmake file of lowlevel library
    set(_cmake_file_lllibrary ${VIDSDK_LOWLEVELLIBRARY_PATH}/lib/cmake/${_lllib}.cmake)
    # check if file is present
    if(EXISTS ${_cmake_file_lllibrary})
      # include cmake file of lowlevel library
      include(${_cmake_file_lllibrary})
    else()
      message(WARNING "Lowlevel library ${_lllib} not found at ${_cmake_file_lllibrary}. Needed cmake variables might be missing")
    endif()
  endforeach(_lllib)
endif()

# include macros to integrate SDK specific files with cmake host targets
include(${VIDSDK_CMAKE_MODULE_DIR}/host.cmake)

###
# videantis SDK cmake module options
#
# VIDSDK_SIM_STDOUT: If set to ON, the stdout output of the simulation will be shown
#                    on stdout of the build system and also redirected to a file
# VIDSDK_HOST_STDOUT: If set to ON, the stdout output of the host application will be shown
#                     on stdout of the build system and also redirected to a file
###
option(VIDSDK_SIM_STDOUT "videantis simulator output is visible on stdout" OFF)
option(VIDSDK_HOST_STDOUT "Host application output is visible on stdout" OFF)

# include internal macros used for generic tasks
include(${VIDSDK_CMAKE_MODULE_DIR}/generic.cmake)
# include macros used to enable registering components
include(${VIDSDK_CMAKE_MODULE_DIR}/register.cmake)
# include macros used to enable building videantis targets
include(${VIDSDK_CMAKE_MODULE_DIR}/build.cmake)
# check if macros for registering v-SP components and
# building videantis v-SP targets are required
if(${VIDSDK_HAS_VSP_TOOLCHAIN})
  # include macros used to enable registering v-SP components
  include(${VIDSDK_CMAKE_MODULE_DIR}/register_vsp.cmake)
  # include macros used to enable building videantis v-SP targets
  include(${VIDSDK_CMAKE_MODULE_DIR}/build_vsp.cmake)
endif()
# include macros used to enable running videantis targets
include(${VIDSDK_CMAKE_MODULE_DIR}/run.cmake)
