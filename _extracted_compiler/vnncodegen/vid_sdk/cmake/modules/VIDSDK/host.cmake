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
# FILENAME:    host.cmake
#
# DESCRIPTION: Support file for videantis SDK to integrate SDK specific files
#              with cmake host targets (C/C++)
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

#[=======================================================================[.rst:
.. cmake:command:: vid_target_add_lllib

  Macro for adding videantis lowlevel library to a cmake (c/c++) target.

  .. code-block:: cmake

    vid_target_add_lllib(target lllib [STATIC])

  Function parameters:

  ``target``
    name of cmake target
  ``lllib``
    lowlevel library to use
  ``STATIC``
    link static lowlevel library
#]=======================================================================]
macro(vid_target_add_lllib _target _lllib)
  cmake_parse_arguments(_ARG "STATIC" "" "" ${ARGN})

  # check if target exists
  if(NOT TARGET ${_target})
    message(FATAL_ERROR "Target ${_target} is not available. This target needs to be generic cmake target for videantis host applications")
  endif()

  # check if requested lowlevel library is available in the used SDK
  if(NOT ${_lllib} IN_LIST VIDSDK_LOWLEVELLIBRARY)
    message(FATAL_ERROR "${_lllib} is not available in ${VIDSDK}")
  endif()

  # get lowlevel library in upper case letters
  string(TOUPPER ${_lllib} _lllib_upper)

  # check if lowlevel library include dirs are defined
  if(NOT DEFINED VID_${_lllib_upper}_HOST_INCLUDE_DIRS)
    message(FATAL_ERROR "Cmake variable VID_${_lllib_upper}_HOST_INCLUDE_DIRS is not defined.")
  endif()

  # check if lowlevel library library dir is defined
  if(NOT DEFINED VID_${_lllib_upper}_HOST_LIB_DIR)
    message(FATAL_ERROR "Cmake variable VID_${_lllib_upper}_HOST_LIB_DIR is not defined.")
  endif()

  # check if libraries to link for lowlevel library are defined
  if(NOT DEFINED VID_${_lllib_upper}_HOST_LIBS)
    message(FATAL_ERROR "Cmake variable VID_${_lllib_upper}_HOST_LIBS is not defined.")
  endif()

  if(NOT "${VID_${_lllib_upper}_HOST_ARCHITECTURE}" STREQUAL "${CMAKE_SYSTEM_PROCESSOR}")
    message(FATAL_ERROR "Low level library is compiled for a ${VID_${_lllib_upper}_HOST_ARCHITECTURE} system.")
  endif()

  # add include directory to target
  target_include_directories(${_target} PRIVATE ${VID_${_lllib_upper}_HOST_INCLUDE_DIRS})
  # link static lowlevel lib
  if (_ARG_STATIC)
    # get clib name of lowlevel lib
    string(REPLACE "lib" "" _clib ${_lllib})
    # copy host libs list
    set(_libs ${VID_${_lllib_upper}_HOST_LIBS})
    # remove lowlevel lib from list
    list(REMOVE_ITEM _libs "${_clib}")
    # set static lowlevel library with full path
    set(_lllib_static ${VID_${_lllib_upper}_HOST_LIB_DIR}/${_lllib}.a)
    # add libraries to link to target (related to lowlevel library)
    target_link_libraries(${_target} PRIVATE ${_lllib_static} ${_libs})
  # link shared lowlevel lib
  else()
    # add directory to find lowlevel library when linking
    target_link_directories(${_target} PRIVATE ${VID_${_lllib_upper}_HOST_LIB_DIR})
    # add libraries to link to target (related to lowlevel library)
    target_link_libraries(${_target} PRIVATE ${VID_${_lllib_upper}_HOST_LIBS})
  endif()
endmacro(vid_target_add_lllib)

#[=======================================================================[.rst:
.. cmake:command:: vid_target_add_binary_header

  Macro for adding videantis header files with binary code to a cmake (c/c++) target.
  The targets defined via TARGETS need to have BIN2H argument set and need to be previously
  defined.

  .. code-block:: cmake

    vid_target_add_binary_header(target TARGETS target1 [target2 ...])

  Function parameters:

  ``target``
    name of cmake target
  ``TARGETS``
    define one or more videantis build targets with binary header file

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VID_BUILD_TARGETS``
    list of videantis build targets of added binary headers

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
#]=======================================================================]
macro(vid_target_add_binary_header _target)
  cmake_parse_arguments(_ARG "" ""
    "TARGETS" ${ARGN})

  # check if target exists
  if(NOT TARGET ${_target})
    message(FATAL_ERROR "Target ${_target} is not available. This target needs to be generic cmake target for videantis host applications")
  endif()

  # check if videantis build targets are defined
  if(NOT DEFINED _ARG_TARGETS)
    message(FATAL_ERROR "vid_target_add_binary_header() needs at least one videantis build target defined via TARGETS")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_target_add_binary_header() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # set target with upper case letters
  string(TOUPPER ${_target} _target_upper)

  # iterate over TARGETS
  foreach(_build_target ${_ARG_TARGETS})
    # set build target with upper case
    string(TOUPPER ${_build_target} _build_target_upper)
    # check if build target for cmake target is available
    get_property(_build_target_property GLOBAL PROPERTY VID_TARGET_${_build_target_upper})
    if(NOT "${_build_target_property}" STREQUAL ${_build_target})
      message(FATAL_ERROR "Target ${_build_target} is not found in videantis build context")
    endif()

    # check if build target has created a binary header file
    get_property(_build_target_bin2h_inc_dir GLOBAL PROPERTY VID_TARGET_${_build_target_upper}_BIN2H_INC_DIR)
    if(NOT DEFINED _build_target_bin2h_inc_dir)
      message(FATAL_ERROR "Could not find binary header include directory for videantis build target ${_build_target}. "
        "Please ensure macro vid_build_target(_vmp/_vsp)() is called with option BIN2H set")
    endif()

    # add include directory to target
    target_include_directories(${_target} PRIVATE ${_build_target_bin2h_inc_dir})

    # add dependency for target to the binary header
    add_dependencies(${_target} binary_header_${_build_target})

    # add videantis build target to list of build targets for this target
    list(APPEND VID_TARGET_${_target_upper}_VID_BUILD_TARGETS ${_build_target})
  endforeach(_build_target)
endmacro(vid_target_add_binary_header)
