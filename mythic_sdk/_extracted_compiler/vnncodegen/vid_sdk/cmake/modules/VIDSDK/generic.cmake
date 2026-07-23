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
# FILENAME:    generic.cmake
#
# DESCRIPTION: Support file for videantis SDK for internal generic macros
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++


#[=======================================================================[.rst:
.. cmake:command:: vid_generate_target_consts

  Helper macro to generate target constant cmake variables.
  This macro is called by :cmake:command:`vid_define_system` and
  :cmake:command:`vid_init_run_target`. It can be also called manually,
  if these variables are missing in the scope of a cmake file.

  .. code-block:: cmake

    vid_generate_target_consts()

  Exposed variables:

  ``VID_TARGET_OCSRAM_ADDR``
    address of the OCSRAM
  ``VID_TARGET_OCSRAM_SIZE``
    size of the OCSRAM
  ``VID_TARGET_SDRAM_BOOT_ADDR``
    address of the SDRAM boot firmware address space
  ``VID_TARGET_SDRAM_BOOT_SIZE``
    size of the SDRAM boot firmware address space
  ``VID_TARGET_SDRAM_OVL_ADDR``
    address of the SDRAM overlay address space
  ``VID_TARGET_SDRAM_OVL_SIZE``
    size of the SDRAM overlay address space
  ``VID_TARGET_DEBUGIF_ADDR``
    address of the debug interface
  ``VID_TARGET_SDRAM_BOOT_MP<ID>``
    SDRAM address for boot firmware of a specific v-MP core
  ``VID_TARGET_SDRAM_BOOT_SP<ID>``
    SDRAM address for boot firmware of a specific v-SP core (variable only exposed, if v-SP tool chain is available)

  Placeholder:

  ``<ID>``
    id of a target core
#]=======================================================================]
macro(vid_generate_target_consts)
  # get global defined lowlevel library
  unset(_lllib)
  get_property(_lllib GLOBAL PROPERTY VID_TARGET_LLLIB)

  # check if lowlevel library was set
  if(NOT DEFINED _lllib)
    message(FATAL_ERROR "No videantis lowlevel library is specified! "
      "Define a project wide lowlevel library with vid_define_system(_vmp/_vsp)().")
  endif()

  # check if requested lowlevel library is available in the used SDK
  if(NOT ${_lllib} IN_LIST VIDSDK_LOWLEVELLIBRARY)
    message(FATAL_ERROR "${_lllib} is not available in ${VIDSDK}")
  endif()

  # get lowlevel library in upper case letters
  string(TOUPPER ${_lllib} _lllib_upper)

  # get all required target constant values from global properties
  foreach(_target_const ${VIDSDK_VLL_TARGET_CONSTS_MAP})
    # get target constant identifier in all lower case letters
    STRING(TOLOWER ${_target_const} _target_const_lower)
    # get value of target constant
    unset(_${_target_const_lower})
    get_property(_${_target_const_lower} GLOBAL PROPERTY VID_TARGET_${_lllib_upper}_${_target_const})
    # check if target constant was set
    if(NOT DEFINED _${_target_const_lower})
      list(FIND VIDSDK_VLL_TARGET_CONSTS_MAP ${_target_const} _map_idx)
      list(GET VIDSDK_VLL_TARGET_CONSTS ${_map_idx} _vll_target_const)
      message(FATAL_ERROR "Target constant ${_vll_target_const}/${_target_const} could not be found!")
    endif()
  endforeach(_target_const)

  # define VID_TARGET_OCSRAM_ADDR
  math(EXPR VID_TARGET_OCSRAM_ADDR
    "${_ocsram_addr}"
    OUTPUT_FORMAT HEXADECIMAL)

  # define VID_TARGET_OCSRAM_SIZE
  math(EXPR VID_TARGET_OCSRAM_SIZE
    "${_ocsram_size}"
    OUTPUT_FORMAT HEXADECIMAL)

  # define VID_TARGET_SDRAM_BOOT_ADDR
  math(EXPR VID_TARGET_SDRAM_BOOT_ADDR
    "${_sdram_addr}"
    OUTPUT_FORMAT HEXADECIMAL)

  if(VIDSDK_HAS_VSP_TOOLCHAIN)
    # calculate VID_TARGET_SDRAM_BOOT_SIZE with v-SPs
    math(EXPR VID_TARGET_SDRAM_BOOT_SIZE
      "${VIDSDK_VLL_NUM_MP} * ${VIDSDK_VLL_MP_FW_SIZE} + ${VIDSDK_VLL_NUM_SP}* ${VIDSDK_VLL_SP_FW_SIZE}"
      OUTPUT_FORMAT HEXADECIMAL)
  else()
      # calculate VID_TARGET_SDRAM_BOOT_SIZE without v-SPs
      math(EXPR VID_TARGET_SDRAM_BOOT_SIZE
        "${VIDSDK_VLL_NUM_MP} * ${VIDSDK_VLL_MP_FW_SIZE}"
        OUTPUT_FORMAT HEXADECIMAL)
  endif()

  # calculate VID_TARGET_SDRAM_OVL_ADDR
  math(EXPR VID_TARGET_SDRAM_OVL_ADDR
    "${_sdram_addr} + ${VID_TARGET_SDRAM_BOOT_SIZE}"
    OUTPUT_FORMAT HEXADECIMAL)

  # calculate VID_TARGET_SDRAM_OVL_SIZE
  math(EXPR VID_TARGET_SDRAM_OVL_SIZE
    "${_sdram_size} - ${VID_TARGET_SDRAM_BOOT_SIZE}"
    OUTPUT_FORMAT HEXADECIMAL)

  # define VID_TARGET_DEBUGIF_ADDR
  math(EXPR VID_TARGET_DEBUGIF_ADDR
    "${_debugif_addr}"
    OUTPUT_FORMAT HEXADECIMAL)

  # define variables to iterate over num of v-MPs to calculate boot addresses
  set(_mp_base ${VID_TARGET_SDRAM_BOOT_ADDR})
  math(EXPR _last_mp_idx "${VIDSDK_VLL_NUM_MP} - 1")

  # iterate over every v-MP index
  foreach(_mp RANGE ${_last_mp_idx})
    # calculate VID_TARGET_SDRAM_BOOT_MPX
    math(EXPR VID_TARGET_SDRAM_BOOT_MP${_mp}
      "${_mp_base} + ${_mp} * ${VIDSDK_VLL_MP_FW_SIZE}"
      OUTPUT_FORMAT HEXADECIMAL)
  endforeach(_mp)

  if(VIDSDK_HAS_VSP_TOOLCHAIN)
    # define variables to iterate over num of v-SPs to calculate boot addresses
    math(EXPR _sp_base
      "${VID_TARGET_SDRAM_BOOT_ADDR} + ${VIDSDK_VLL_NUM_MP} * ${VIDSDK_VLL_MP_FW_SIZE}"
      OUTPUT_FORMAT HEXADECIMAL)
    math(EXPR _last_sp_idx "${VIDSDK_VLL_NUM_SP} - 1")

    # iterate over every v-MP index
    foreach(_sp RANGE ${_last_sp_idx})
      # calculate VID_TARGET_SDRAM_BOOT_SPX
      math(EXPR VID_TARGET_SDRAM_BOOT_SP${_sp}
        "${_sp_base} + ${_sp} * ${VIDSDK_VLL_SP_FW_SIZE}"
        OUTPUT_FORMAT HEXADECIMAL)
    endforeach(_sp)
  endif()
endmacro(vid_generate_target_consts)

#[=======================================================================[.rst:
.. cmake:command:: vid_define_system

  Helper macro to define a global system with lowlevel library, target cpu
  and target soc. It parses the header file with target constants of the
  specific lowlevel library and exposes the related variables as cmake
  variables. This marco can be define a system only for one core type
  at the time. Call it twice to define system for v-SP and v-MP.

  .. code-block:: cmake

    vid_define_system(LLLIB lllib TARGET_CPU target_cpu TARGET_SOC target_soc [SP/MP]
        [NO_WARN_OVERWRITE])

  Function parameters:

  ``LLLIB``
    lowlevel library
  ``TARGET_CPU``
    target cpu
  ``TARGET_SOC``
    target soc
  ``SP``
    define system for v-SP (if v-SP tool chain is available)
  ``MP``
    define system for v-MP (default, if SP or MP is not set)
  ``NO_WARN_OVERWRITE``
    silence warnings when overwriting a previously defined system

  Exposed global properties:

  ``VID_TARGET_LLIB``
    videantis lowlevel library
  ``VID_TARGET_TARGET_CPU_<TYPE>``
    cpu of the target (e.g. mp4.0)
  ``VID_TARGET_TARGET_SOC_<TYPE>``
    SoC of the target
  ``VID_TARGET_<LLLIB>_SDRAM_ADDR``
    SDRAM start address parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_SDRAM_SIZE``
    SDRAM size parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_OCSRAM_ADDR``
    OCSRAM start address parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_OCSRAM_SIZE``
    OCSRAM size parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_DEBUGIF_ADDR``
    Debug interface start address parsed from target_consts.h of selected lowlevel library

  Exposed variables:

  ``VID_TARGET_OCSRAM_ADDR``
    address of the OCSRAM
  ``VID_TARGET_OCSRAM_SIZE``
    size of the OCSRAM
  ``VID_TARGET_SDRAM_BOOT_ADDR``
    address of the SDRAM boot firmware address space
  ``VID_TARGET_SDRAM_BOOT_SIZE``
    size of the SDRAM boot firmware address space
  ``VID_TARGET_SDRAM_OVL_ADDR``
    address of the SDRAM overlay address space
  ``VID_TARGET_SDRAM_OVL_SIZE``
    size of the SDRAM overlay address space
  ``VID_TARGET_DEBUGIF_ADDR``
    address of the debug interface
  ``VID_TARGET_SDRAM_BOOT_MP<ID>``
    SDRAM address for boot firmware of a specific v-MP core
  ``VID_TARGET_SDRAM_BOOT_SP<ID>``
    SDRAM address for boot firmware of a specific v-SP core (variable only exposed, if v-SP tool chain is available)

  Placeholder:

  ``<TYPE>``
    type of target SP or MP in all upper case letters
  ``<LLLIB>``
    lowlevel library in all upper case letters
  ``<ID>``
    id of a target core
#]=======================================================================]
macro(vid_define_system)
  set(_options NO_WARN_OVERWRITE MP)
  # check if macros for v-SP targets are required
  if(${VIDSDK_HAS_VSP_TOOLCHAIN})
    list(APPEND _options SP)
  endif()

  cmake_parse_arguments(_ARG "${_options}"
    "LLLIB;TARGET_CPU;TARGET_SOC" "" ${ARGN})

  # check if LLLIB is defined
  if(NOT DEFINED _ARG_LLLIB)
    message(FATAL_ERROR "LLLIB is a required argument when calling vid_define_system()")
  endif()

  # check if TARGET_CPU is defined
  if(NOT DEFINED _ARG_TARGET_CPU)
    message(FATAL_ERROR "TARGET_CPU is a required argument when calling vid_define_system()")
  endif()

  # check if TARGET_SOC is defined
  if(NOT DEFINED _ARG_TARGET_SOC)
    message(FATAL_ERROR "TARGET_SOC is a required argument when calling vid_define_system()")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_define_system() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # if v-SP tool chain is available ensure that only MP or SP argument is set
  if(${VIDSDK_HAS_VSP_TOOLCHAIN})
    if(${_ARG_MP} AND ${_ARG_SP})
      message(FATAL_ERROR "vid_define_system() can be called only with option SP or MP")
    endif()
  endif()

  # set default type to v-MP
  set(_type MP)
  # if SP option is set, set type to v-SP
  if(${_ARG_SP})
    set(_type SP)
  endif()

  # check if requested lowlevel library is available in the used SDK
  if(NOT ${_ARG_LLLIB} IN_LIST VIDSDK_LOWLEVELLIBRARY)
    message(FATAL_ERROR "${_ARG_LLLIB} is not available in ${VIDSDK}")
  endif()
  get_property(_global_lllib GLOBAL PROPERTY VID_TARGET_LLLIB)
  if((DEFINED _global_lllib) AND (NOT ${_global_lllib} STREQUAL ${_ARG_LLLIB}) AND (NOT ${_ARG_NO_WARN_OVERWRITE}))
    message(WARNING "A project wide lowlevel library is already defined: "
      "Overwriting ${_global_lllib} to ${_ARG_LLLIB}")
  endif()
  set_property(GLOBAL PROPERTY VID_TARGET_LLLIB ${_ARG_LLLIB})

  get_property(_global_target_cpu GLOBAL PROPERTY VID_TARGET_TARGET_CPU_${_type})
  if((DEFINED _global_target_cpu) AND (NOT ${_ARG_NO_WARN_OVERWRITE}))
    message(WARNING "A project wide target cpu for v-${_type} is already defined: "
      "Overwriting ${_global_target_cpu} to ${_ARG_TARGET_CPU}")
  endif()
  set_property(GLOBAL PROPERTY VID_TARGET_TARGET_CPU_${_type} ${_ARG_TARGET_CPU})

  get_property(_global_target_soc GLOBAL PROPERTY VID_TARGET_TARGET_SOC_${_type})
  if((DEFINED _global_target_soc) AND (NOT ${_ARG_NO_WARN_OVERWRITE}))
    message(WARNING "A project wide target soc for v-${_type} is already defined: "
      "Overwriting ${_global_target_soc} to ${_ARG_TARGET_SOC}")
  endif()
  set_property(GLOBAL PROPERTY VID_TARGET_TARGET_SOC_${_type} ${_ARG_TARGET_SOC})

  # lowlevel library in all upper case letters
  string(TOUPPER ${_ARG_LLLIB} _lllib_upper)

  # check if lowlevel library include dirs are defined
  if(NOT DEFINED VID_${_lllib_upper}_HOST_INCLUDE_DIRS)
    message(FATAL_ERROR "Cmake variable VID_${_lllib_upper}_HOST_INCLUDE_DIRS is not defined.")
  endif()

  # define full path to target_consts.h file
  set(_target_consts_file ${VID_${_lllib_upper}_HOST_INCLUDE_DIRS}/target_consts.h)

  # check if target_consts.h file exists
  if(NOT EXISTS ${_target_consts_file})
    message(FATAL_ERROR "Could not find target_consts.h file in directory ${VID_${_lllib_upper}_HOST_INCLUDE_DIRS}")
  endif()

  # add target_consts.h file to CMAKE_CONFIGURE_DEPENDS list to ensure cmake reconfigures, if content of the file changes
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_target_consts_file})

  # remove comments, empty lines and multiple whitespace characters from target_consts.h using gcc compiler
  execute_process(COMMAND ${VMPCC_EXECUTABLE} --preprocess -dD -E -P -Wno-unused-command-line-argument ${_target_consts_file}
    OUTPUT_VARIABLE _target_consts)

  # remove trailing empty row
  string(REGEX REPLACE "\n[ \t]*$" "" _target_consts "${_target_consts}")
  # replace all new lines with semicolon to generate cmake list
  string(REPLACE "\n" ";" _target_consts "${_target_consts}")

  # initialize a counter to check target consts parsed
  set(_target_consts_counter 0)
  # iterate over each line of target_consts.h
  foreach(_line ${_target_consts})
    # remove opening bracket from line
    string(REPLACE "(" "" _line "${_line}")
    # remove closing bracket from line
    string(REPLACE ")" "" _line "${_line}")
    # strip line
    string(STRIP ${_line} _line)
    # replace multiple spaces with one space
    string(REGEX REPLACE "[ ]+" " " _line "${_line}")
    # replace whitespace with semicolon to make a list
    string(REPLACE " " ";" _list_line ${_line})
    # get length of list
    list(LENGTH _list_line _list_length)

    # ensure the line contains at least 3 list elements (#define, NAME, VALUE)
    if(${_list_length} GREATER_EQUAL 3)
      # get the name of the define in target_consts.h (cmake list position 1)
      list(GET _list_line 1 _var_name)
      # check if variable is required to be parsed
      if(${_var_name} IN_LIST VIDSDK_VLL_TARGET_CONSTS)
          # get the value of the define in target_consts.h (cmake list position 2)
          list(GET _list_line 2 _var_value)
          # remove ull suffix from parsed constant
          string(REPLACE "ull" "" _var_value ${_var_value})
          # remove u suffix from parsed constant
          string(REPLACE "u" "" _var_value ${_var_value})

          # get the variable index from the list that contains the target consts to be parsed
          list(FIND VIDSDK_VLL_TARGET_CONSTS ${_var_name} _var_idx)
          # get the mapped name for the target constant for cmake
          list(GET VIDSDK_VLL_TARGET_CONSTS_MAP ${_var_idx} _map_var)

          # set global property for this variable
          set_property(GLOBAL PROPERTY VID_TARGET_${_lllib_upper}_${_map_var} ${_var_value})

          # increment target consts counter
          math(EXPR _target_consts_counter "${_target_consts_counter} + 1")
      endif()
    endif()
  endforeach(_line)

  # get number of required target constants
  list(LENGTH VIDSDK_VLL_TARGET_CONSTS _vidsdk_vll_target_consts_length)
  # check if all target constants have been found
  if(NOT ${_target_consts_counter} EQUAL ${_vidsdk_vll_target_consts_length})
    message(FATAL_ERROR "Not all required target constants (${VIDSDK_VLL_TARGET_CONSTS}) could be parsed from ${_target_consts_file}!")
  endif()

  # generate target consts cmake variables
  vid_generate_target_consts()
endmacro(vid_define_system)

#[=======================================================================[.rst:
.. cmake:command:: vid_define_system_vmp

  Helper macro to define a global system for v-MP with lowlevel library,
  target cpu and target soc. It parses the header file with target constants
  of the specific lowlevel library and exposes the related variables as
  cmake variables.
  This is a wrapper macro for :cmake:command:`vid_define_system`.

  .. code-block:: cmake

    vid_define_system_vmp(LLLIB lllib TARGET_CPU target_cpu TARGET_SOC target_soc
        [NO_WARN_OVERWRITE])

  Function parameters:

  ``LLLIB``
    lowlevel library
  ``TARGET_CPU``
    target cpu
  ``TARGET_SOC``
    target soc
  ``NO_WARN_OVERWRITE``
    silence warnings when overwriting a previously defined system

  Exposed global properties:

  ``VID_TARGET_LLIB``
    videantis lowlevel library
  ``VID_TARGET_TARGET_CPU_MP``
    cpu of the target (e.g. mp4.0)
  ``VID_TARGET_TARGET_SOC_MP``
    SoC of the target
  ``VID_TARGET_<LLLIB>_SDRAM_ADDR``
    SDRAM start address parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_SDRAM_SIZE``
    SDRAM size parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_OCSRAM_ADDR``
    OCSRAM start address parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_OCSRAM_SIZE``
    OCSRAM size parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_DEBUGIF_ADDR``
    Debug interface start address parsed from target_consts.h of selected lowlevel library

  Exposed variables:

  ``VID_TARGET_OCSRAM_ADDR``
    address of the OCSRAM
  ``VID_TARGET_OCSRAM_SIZE``
    size of the OCSRAM
  ``VID_TARGET_SDRAM_BOOT_ADDR``
    address of the SDRAM boot firmware address space
  ``VID_TARGET_SDRAM_BOOT_SIZE``
    size of the SDRAM boot firmware address space
  ``VID_TARGET_SDRAM_OVL_ADDR``
    address of the SDRAM overlay address space
  ``VID_TARGET_SDRAM_OVL_SIZE``
    size of the SDRAM overlay address space
  ``VID_TARGET_DEBUGIF_ADDR``
    address of the debug interface
  ``VID_TARGET_SDRAM_BOOT_MP<ID>``
    SDRAM address for boot firmware of a specific v-MP core
  ``VID_TARGET_SDRAM_BOOT_SP<ID>``
    SDRAM address for boot firmware of a specific v-SP core (variable only exposed, if v-SP tool chain is available)

  Placeholder:

  ``<LLLIB>``
    lowlevel library in all upper case letters
  ``<ID>``
    id of a target core
#]=======================================================================]
macro(vid_define_system_vmp)
  cmake_parse_arguments(_ARG "NO_WARN_OVERWRITE"
    "LLLIB;TARGET_CPU;TARGET_SOC" "" ${ARGN})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_define_system_vmp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # unset no_warn_overwrite from possible prior calls of vid_define_system_vmp()
  unset(_no_warn_overwrite)
  # check if NO_WARN_OVERWRITE is set
  if(${_ARG_NO_WARN_OVERWRITE})
    set(_no_warn_overwrite NO_WARN_OVERWRITE)
  endif()

  vid_define_system(LLLIB ${_ARG_LLLIB} TARGET_CPU ${_ARG_TARGET_CPU} TARGET_SOC ${_ARG_TARGET_SOC} MP ${_no_warn_overwrite})
endmacro(vid_define_system_vmp)


#[=======================================================================[.rst:
.. cmake:command:: vid_define_system_vsp

  Helper macro to define a global system for v-SP with lowlevel library,
  target cpu and target soc. It parses the header file with target constants
  of the specific lowlevel library and exposes the related variables as
  cmake variables.
  This is a wrapper macro for :cmake:command:`vid_define_system`.

  .. code-block:: cmake

    vid_define_system_vsp(LLLIB lllib TARGET_CPU target_cpu TARGET_SOC target_soc
        [NO_WARN_OVERWRITE])

  Function parameters:

  ``LLLIB``
    lowlevel library
  ``TARGET_CPU``
    target cpu
  ``TARGET_SOC``
    target soc
  ``NO_WARN_OVERWRITE``
    silence warnings when overwriting a previously defined system

  Exposed global properties:

  ``VID_TARGET_LLIB``
    videantis lowlevel library
  ``VID_TARGET_TARGET_CPU_SP``
    cpu of the target (e.g. sp3.1)
  ``VID_TARGET_TARGET_SOC_SP``
    SoC of the target
  ``VID_TARGET_<LLLIB>_SDRAM_ADDR``
    SDRAM start address parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_SDRAM_SIZE``
    SDRAM size parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_OCSRAM_ADDR``
    OCSRAM start address parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_OCSRAM_SIZE``
    OCSRAM size parsed from target_consts.h of selected lowlevel library
  ``VID_TARGET_<LLLIB>_DEBUGIF_ADDR``
    Debug interface start address parsed from target_consts.h of selected lowlevel library

  Exposed variables:

  ``VID_TARGET_OCSRAM_ADDR``
    address of the OCSRAM
  ``VID_TARGET_OCSRAM_SIZE``
    size of the OCSRAM
  ``VID_TARGET_SDRAM_BOOT_ADDR``
    address of the SDRAM boot firmware address space
  ``VID_TARGET_SDRAM_BOOT_SIZE``
    size of the SDRAM boot firmware address space
  ``VID_TARGET_SDRAM_OVL_ADDR``
    address of the SDRAM overlay address space
  ``VID_TARGET_SDRAM_OVL_SIZE``
    size of the SDRAM overlay address space
  ``VID_TARGET_DEBUGIF_ADDR``
    address of the debug interface
  ``VID_TARGET_SDRAM_BOOT_MP<ID>``
    SDRAM address for boot firmware of a specific v-MP core
  ``VID_TARGET_SDRAM_BOOT_SP<ID>``
    SDRAM address for boot firmware of a specific v-SP core (variable only exposed, if v-SP tool chain is available)

  Placeholder:

  ``<LLLIB>``
    lowlevel library in all upper case letters
  ``<ID>``
    id of a target core
#]=======================================================================]
if(${VIDSDK_HAS_VSP_TOOLCHAIN})
  macro(vid_define_system_vsp)
    cmake_parse_arguments(_ARG "NO_WARN_OVERWRITE"
      "LLLIB;TARGET_CPU;TARGET_SOC" "" ${ARGN})

    # raise a warning for the case some arguments cannot parsed
    if(DEFINED _ARG_UNPARSED_ARGUMENTS)
      message(WARNING "vid_define_system_vsp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
    endif()

    # unset no_warn_overwrite from possible prior calls of vid_define_system_vsp()
    unset(_no_warn_overwrite)
    # check if NO_WARN_OVERWRITE is set
    if(${_ARG_NO_WARN_OVERWRITE})
      set(_no_warn_overwrite NO_WARN_OVERWRITE)
    endif()

    vid_define_system(LLLIB ${_ARG_LLLIB} TARGET_CPU ${_ARG_TARGET_CPU} TARGET_SOC ${_ARG_TARGET_SOC} SP ${_no_warn_overwrite})
  endmacro(vid_define_system_vsp)
endif()

#[=======================================================================[.rst:
.. cmake:command:: vid_parse_simcfg_as_list

  Helper macro to parse a videantis simulator config from a file or
  a string as a cmake list.
  This macro is intended for internal.

  .. code-block:: cmake

    vid_parse_simcfg_as_list([FROM_FILE simcfg] [FROM_STRING simcfg])

  Function parameters:

  ``FROM_FILE``
    simulator config file
  ``FROM_STRING``
    simulator config string
  ``OUTPUT``
    _SIMCFG_LIST
#]=======================================================================]
macro(vid_parse_simcfg_as_list)
  cmake_parse_arguments(_ARG ""
    "FROM_FILE;FROM_STRING" "" ${ARGN})

  if((NOT DEFINED _ARG_FROM_FILE) AND (NOT DEFINED _ARG_FROM_STRING))
    message(FATAL_ERROR "vid_parse_simcfg_as_list() requires to set argument FROM_FILE or FROM_STRING")
  endif()

  if((DEFINED _ARG_FROM_FILE) AND (DEFINED _ARG_FROM_STRING))
    message(FATAL_ERROR "vid_parse_simcfg_as_list() requires to set argument FROM_FILE or FROM_STRING not both")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_parse_simcfg_as_list() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  set(_simcfg_cached OFF)
  if(DEFINED _ARG_FROM_FILE)
    # check if the simcfg file exists
    if(NOT EXISTS ${_ARG_FROM_FILE})
      message(FATAL_ERROR "The provided simcfg file ${_ARG_FROM_FILE} does not exist")
    endif()

    # check if the result for this file is already cached (avoids re-parsing the
    # same template file thousands of times during configuration of large test suites)
    string(MAKE_C_IDENTIFIER "${_ARG_FROM_FILE}" _simcfg_cache_key)
    get_property(_simcfg_cached GLOBAL PROPERTY _VID_SIMCFG_CACHE_${_simcfg_cache_key} SET)
    if(_simcfg_cached)
      get_property(_SIMCFG_LIST GLOBAL PROPERTY _VID_SIMCFG_CACHE_${_simcfg_cache_key})
    else()
      # read file content
      file(READ ${_ARG_FROM_FILE} _SIMCFG_LIST)
    endif()
  endif()

  # put string into _SIMCFG_LIST variable for further processing
  if(DEFINED _ARG_FROM_STRING)
    set(_SIMCFG_LIST "${_ARG_FROM_STRING}")
  endif()

  if(NOT _simcfg_cached)
    # convert simcfg template to cmake list
    # remove comments (replaces execute_process+sed)
    string(REGEX REPLACE "#[^\n]*" "" _SIMCFG_LIST "${_SIMCFG_LIST}")
    # replace all new lines with spaces
    string(REPLACE "\n" " " _SIMCFG_LIST "${_SIMCFG_LIST}")
    # replace multiple tabs or spaces with one space
    string(REGEX REPLACE "[ \t]+" " " _SIMCFG_LIST "${_SIMCFG_LIST}")
    # strip string to remove leading and trailing whitespace
    string(STRIP "${_SIMCFG_LIST}" _SIMCFG_LIST)
    # replace whitespace with semicolon to generate final cmake list
    string(REPLACE " " ";" _SIMCFG_LIST "${_SIMCFG_LIST}")
    if(DEFINED _ARG_FROM_FILE)
      # store result in global cache for subsequent calls with the same file
      set_property(GLOBAL PROPERTY _VID_SIMCFG_CACHE_${_simcfg_cache_key} "${_SIMCFG_LIST}")
    endif()
  endif()
endmacro(vid_parse_simcfg_as_list)

#[=======================================================================[.rst:
.. cmake:command:: vid_cached_configure_file

  Helper macro for a cached version of configure_file() to avoid
  repetitive reads of the same template file. The configured file
  is additionally as cmake string available.
  This macro is intended for internal.

  .. code-block:: cmake

    vid_cached_configure_file(input_file output_file)

  Function parameters:

  ``input_file``
    input file
  ``output_file``
    output file
  ``OUTPUT``
    _FILE_CONTENT
#]=======================================================================]
macro(vid_cached_configure_file _input_file _output_file)
  # Read template with caching (same template shared by all tests), perform @VAR@ substitution,
  # then defer file write to generate phase via file(GENERATE) to avoid per-test disk I/O at
  # configure time (configure_file writes immediately; file(GENERATE) batches all writes).
  string(MAKE_C_IDENTIFIER "${_input_file}" _tmpl_cache_key)
  # check if template was already read
  get_property(_cached GLOBAL PROPERTY _VID_TMPL_CACHE_${_tmpl_cache_key} SET)
  if(_tmpl_cached)
    get_property(_tmpl_content GLOBAL PROPERTY _VID_TMPL_CACHE_${_tmpl_cache_key})
  else()
    file(READ ${_input_file} _tmpl_content)
    set_property(GLOBAL PROPERTY _VID_TMPL_CACHE_${_tmpl_cache_key} "${_tmpl_content}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_input_file}")
  endif()
  unset(_tmpl_cache_key)
  unset(_tmpl_cached)
  # configure content read from template
  string(CONFIGURE "${_tmpl_content}" _FILE_CONTENT @ONLY)
  unset(_tmpl_content)
  # defer the file write to the generate phase (no disk I/O per test during configure)
  file(GENERATE OUTPUT ${_output_file} CONTENT "${_FILE_CONTENT}")
endmacro(vid_cached_configure_file)

#[=======================================================================[.rst:
.. cmake:command:: vid_get_arg_from_simcfg_list

  Helper macro get an argument from a simcfg list.
  This macro is intended for internal use.

  .. code-block:: cmake

    vid_get_arg_from_simcfg_list(argument SIMCFG simcfg [SUB sub] [AFTER after]
        [POS pos] [REQUIRED] [MULTIPLE] [IS_SET])

  Function parameters:

  ``argument``
    argument to get from a parsed simulator config (e.g. -cores)
  ``SIMCFG``
    list of simcfg parameter
  ``SUB``
    searches for a sub argument (e.g. div returns X for div=X)
  ``AFTER``
    searches for the next value after a (regex) pattern (e.g mp returns X for "mp4.0 X")
  ``POS``
    return a sub argument at defined position (position indexing starts with 0)
  ``REQUIRED``
    raise an error when argument is not found
  ``MULTIPLE``
    searches full simcfg config and returns a list if search result is present multiple times
  ``IS_SET``
    return TRUE if argument is set minimum once in simcfg (if not set: FALSE).
    IS_SET cannot defined in combination with SUB, AFTER and POS.
    Combining with MULTIPLE is not forbidden, but does not change the behavior.
  ``OUTPUT``
    _SIMCFG_ARG

  .. note::

    SUB, AFTER and POS are not allowed to be defined at the same time
#]=======================================================================]
macro(vid_get_arg_from_simcfg_list _argument)
  cmake_parse_arguments(_ARG "IS_SET;MULTIPLE;REQUIRED" "SUB;AFTER;POS"
    "SIMCFG" ${ARGN})

    # check if argument starts with "-"
    if(NOT ${_argument} MATCHES "^-")
      message(FATAL_ERROR "Specified argument \"${_argument}\" for vid_get_arg_from_simcfg_list() needs to start with \"-\"")
    endif()

    if(NOT DEFINED _ARG_SIMCFG)
      message(FATAL_ERROR "SIMCFG is a required parameter when calling vid_get_arg_from_simcfg_list()")
    endif()

    # is_set can be only used with no special search mode defined
    if(_ARG_IS_SET AND ((DEFINED _ARG_SUB) OR (DEFINED _ARG_AFTER) OR (DEFINED _ARG_POS)))
      message(FATAL_ERROR "Checking if a simcfg argument is set, can be only done when no search mode (SUB, POS or AFTER) is defined")
    endif()

    # check if only argument SUB is defined
    if((DEFINED _ARG_SUB) AND ((DEFINED _ARG_AFTER) OR (DEFINED _ARG_POS)))
      message(FATAL_ERROR "Only one search mode (SUB, POS or AFTER) can be defined at the same macro call of vid_get_arg_from_simcfg_list()")
    endif()

    # check if only argument AFTER is defined
    if((DEFINED _ARG_AFTER) AND ((DEFINED _ARG_SUB) OR (DEFINED _ARG_POS)))
      message(FATAL_ERROR "Only one search mode (SUB, POS or AFTER) can be defined at the same macro call of vid_get_arg_from_simcfg_list()")
    endif()

    # check if only argument POS is defined
    if((DEFINED _ARG_POS) AND ((DEFINED _ARG_SUB) OR (DEFINED _ARG_AFTER)))
      message(FATAL_ERROR "Only one search mode (SUB, POS or AFTER) can be defined at the same macro call of vid_get_arg_from_simcfg_list()")
    endif()

    # raise a warning for the case some arguments cannot parsed
    if(DEFINED _ARG_UNPARSED_ARGUMENTS)
      message(WARNING "vid_get_arg_from_simcfg_list() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
    endif()

    # clear output variable
    unset(_SIMCFG_ARG)
    # if checking for an argument set, initialize to FALSE
    if(_ARG_IS_SET)
      set(_SIMCFG_ARG FALSE)
    endif()

    # find next argument after a pattern (regex allowed)
    if(DEFINED _ARG_AFTER)
      # initialize conditional variables for inner loop control
      set(_check_next FALSE)
      set(_get_next FALSE)
      # loop over arguments from simcfg
      foreach(_simcfg_arg ${_ARG_SIMCFG})
        # if specified argument equals actual argument
        if(${_argument} STREQUAL ${_simcfg_arg})
          # check next argument
          set(_check_next TRUE)
          # go to next argument
          continue()
        endif()
        # if next element should be checked
        if(_check_next)
          # if next element is a new parameter
          if(${_simcfg_arg} MATCHES "^-")
            # stop loop, if parsing for multiple arguments is not set
            if(NOT _ARG_MULTIPLE)
              break()
            endif()
            # if specified argument not equals actual argument
            if(NOT ${_argument} STREQUAL ${_simcfg_arg})
              # don't check the next argument
              set(_check_next FALSE)
            endif()
            # go to next argument
            continue()
          endif()
          # if pattern for after matches
          if(${_simcfg_arg} MATCHES ${_ARG_AFTER})
            # get parameter of next argument
            set(_get_next TRUE)
            # go to next argument
            continue()
          endif()
          # if argument should be extracted
          if(_get_next)
            # write value to output
            list(APPEND _SIMCFG_ARG ${_simcfg_arg})
            # stop loop, if parsing for multiple arguments is not set
            if(NOT _ARG_MULTIPLE)
              break()
            endif()
            # don't check and get next item by default
            set(_check_next FALSE)
            set(_get_next FALSE)
          endif()
        endif()
      endforeach(_simcfg_arg)
    # find sub argument
    elseif(DEFINED _ARG_SUB)
      # initialize conditional variables for inner loop control
      set(_check_next FALSE)
      # loop over arguments from simcfg
      foreach(_simcfg_arg ${_ARG_SIMCFG})
        # if specified argument equals actual argument
        if(${_argument} STREQUAL ${_simcfg_arg})
          # check next argument
          set(_check_next TRUE)
          # go to next argument
          continue()
        endif()
        # if next element should be checked
        if(_check_next)
          # if next element is a new parameter
          if(${_simcfg_arg} MATCHES "^-")
            # stop loop, if parsing for multiple arguments is not set
            if(NOT _ARG_MULTIPLE)
              break()
            endif()
            # if specified argument not equals actual argument
            if(NOT ${_argument} STREQUAL ${_simcfg_arg})
              # don't check next argument
              set(_check_next FALSE)
            endif()
            # go to next argument
            continue()
          endif()
          # if sub argument for matches
          if(${_simcfg_arg} MATCHES "^${_ARG_SUB}")
            # delete sub argument prefix from argument
            string(REPLACE "${_ARG_SUB}=" "" _tmp ${_simcfg_arg})
            # write cleared value to output
            list(APPEND _SIMCFG_ARG ${_tmp})
            # stop loop, if parsing for multiple arguments is not set
            if(NOT _ARG_MULTIPLE)
              break()
            endif()
            # don't check next item by default
            set(_check_next FALSE)
          endif()
        endif()
      endforeach(_simcfg_arg)
    elseif(DEFINED _ARG_POS)
      # initialize conditional variables for inner loop control
      set(_check_next FALSE)
      set(_param_pos 0)
      # loop over arguments from simcfg
      foreach(_simcfg_arg ${_ARG_SIMCFG})
        # if specified argument equals actual argument
        if(${_argument} STREQUAL ${_simcfg_arg})
          # check next argument
          set(_check_next TRUE)
          # go to next argument
          continue()
        endif()
        # if next element should be checked
        if(_check_next)
          # if next element is a new parameter
          if(${_simcfg_arg} MATCHES "^-")
            # stop loop, if parsing for multiple arguments is not set
            if(NOT _ARG_MULTIPLE)
              break()
            endif()
            # if specified argument not equals actual argument
            if(NOT ${_argument} STREQUAL ${_simcfg_arg})
              # don't check next argument
              set(_check_next FALSE)
            endif()
            # reset parameter position counter
            set(_param_pos 0)
            # go to next argument
            continue()
          endif()
          # if actual position matches specified position
          if(${_param_pos} EQUAL ${_ARG_POS})
            # write value to output
            list(APPEND _SIMCFG_ARG ${_simcfg_arg})
            # stop loop, if parsing for multiple arguments is not set
            if(NOT _ARG_MULTIPLE)
              break()
            endif()
            # don't check next item by default
            set(_check_next FALSE)
            set(_param_pos 0)
          else()
            # if not actual position matches increment counter
            math(EXPR _param_pos "${_param_pos} + 1")
          endif()
        endif()
      endforeach(_simcfg_arg)
    else()
      # initialize conditional variables for inner loop control
      set(_check_next FALSE)
      set(_quoted FALSE)
      set(_quoted_start FALSE)
      # loop over arguments from simcfg
      foreach(_simcfg_arg ${_ARG_SIMCFG})
        # if specified argument equals actual argument
        if((NOT _check_next) AND (${_argument} STREQUAL ${_simcfg_arg}))
          # if is_set is defined
          if(_ARG_IS_SET)
            # set output to true to indicate that argument is found
            set(_SIMCFG_ARG TRUE)
            # stop loop
            break()
          endif()
          # check next argument
          set(_check_next TRUE)
          # go to next argument
          continue()
        endif()
        # if next element should be checked
        if(_check_next)
          # check if element starts with a double quote
          if(NOT _quoted AND (${_simcfg_arg} MATCHES "^\""))
            set(_quoted TRUE)
            set(_quoted_start TRUE)
          endif()
          # if next element is a new parameter (only if not quoted)
          if(NOT _quoted AND (${_simcfg_arg} MATCHES "^-"))
            # stop loop, if parsing for multiple arguments is not set
            if(NOT _ARG_MULTIPLE)
              break()
            endif()
            # if new parameter is not the defined argument
            if(NOT ${_argument} STREQUAL ${_simcfg_arg})
              # don't check next item by default
              set(_check_next FALSE)
            endif()
          else()
            # check if element ends with a double quote (only if not the first double quote)
            if(NOT _quoted_start AND (${_simcfg_arg} MATCHES "\"$"))
              set(_quoted FALSE)
            endif()
            # remove double quote from element
            string(REPLACE "\"" "" _simcfg_arg ${_simcfg_arg})
            # write value to output
            list(APPEND _SIMCFG_ARG ${_simcfg_arg})
            # remove first double quote
            set(_quoted_start FALSE)
          endif()
        endif()
      endforeach(_simcfg_arg)
    endif()

    # check if this argument was a required argument
    if(_ARG_REQUIRED AND (NOT DEFINED _SIMCFG_ARG))
      if(DEFINED _ARG_AFTER)
        message(FATAL_ERROR "Could not find argument ${_argument} or sub argument based on pattern ${_ARG_AFTER}. Please check argument and/or simulator config")
      elseif(DEFINED _ARG_SUB)
        message(FATAL_ERROR "Could not find argument ${_argument} or find sub argument ${_ARG_SUB}. Please check argument and/or simulator config")
      elseif(DEFINED _ARG_POS)
        message(FATAL_ERROR "Could not find argument ${_argument} or sub argument for position ${_ARG_POS}. Please check argument and/or simulator config")
      else()
        message(FATAL_ERROR "Could not find argument ${_argument}. Please check argument and/or simulator config")
      endif()
    endif()
endmacro(vid_get_arg_from_simcfg_list)

#[=======================================================================[.rst:
.. cmake:command:: vid_check_and_translate_data_memory_vmp

  Helper macro to check the validity of provided data memory and
  translate the data memory into an ID supported by videantis lowlevel
  library.
  This macro is intended for internal use.

  .. code-block:: cmake

    vid_check_and_translate_data_memory_vmp(data_memory_var)

  Function parameters:

  ``data_memory_var``
    variable with a data memory definition
  ``OUTPUT``
    _DATA_MEMORY_ID
#]=======================================================================]
macro(vid_check_and_translate_data_memory_vmp _data_memory_var)
  # list with definitions of available data memories (v-MP 4.0 or greater)
  set(_data_memories_vmp dmem dmem2 dmem3)

  # create a all lower case version of the content of the provided data memory
  string(TOLOWER ${${_data_memory_var}} _data_memory_lower)

  # check if defined data memory is in reference list of data memories
  if(NOT ${_data_memory_lower} IN_LIST _data_memories_vmp)
    message(FATAL_ERROR "Defined data memory in variable ${_data_memory_var} is not valid: ${${_data_memory_var}}")
  endif()

  # get corresponding index of data memory from list
  list(FIND _data_memories_vmp ${_data_memory_lower} _dmem_idx)
  # translate index from data memory list into ID used in lowlevel library
  math(EXPR _DATA_MEMORY_ID "${_dmem_idx} + 1")
endmacro(vid_check_and_translate_data_memory_vmp)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_test

  Macro for adding a test via ctest with videantis build and run targets and additional commands.
  To add a test at least a build target, a run target or test commands need to be defined.
  A test added with the help of this macro doesn't use the build system when a test is executed.
  This means every cmake target required to execute the test needs to be build before hand.
  The exception is a videantis build target added to a test. The tests are stateless and every part of the
  test will be executed again even if it succeeded in a previous run or have been built/run successfully
  with the build system. Additional outputs defined will be added to the clean target of the build system.
  Videantis build and run targets are executed in their working directory including the optional
  pre and post commands for each these targets. It is possible to add more then one videantis build targets
  to a test. In this case all build targets need to have the same working directory. The default ctest
  template ships with the videantis SDK cmake module, but can be replaced to customize test behavior.
  Additionally custom cmake variables can be added to the cmake script call. For every videantis build
  and/or run target used in a test, the macro creates a resource lock to ensure no tests with the same
  target are run in parallel.

  More details about can be found on page :ref:`testing`

  .. note::

    Adding the same build or run target to multiple tests is not recommended. Exclude them from the tests
    and execute them via the build system before starting the tests.

  .. code-block:: cmake

    vid_add_test(name [BUILD_TARGETS target1 [target2 ...] [PRE_BUILD_TARGET_COMMANDS cmd1 [cmd2 ...]]
      [POST_BUILD_TARGET_COMMANDS cmd1 [cmd2 ...]] [ADDITIONAL_BUILD_TARGET_OUTPUTS file1 [file2 ...]]]
      [RUN_TARGET target [PRE_RUN_TARGET_COMMANDS cmd1 [cmd2 ...]]
      [POST_RUN_TARGET_COMMANDS cmd1 [cmd2 ...]] [ADDITIONAL_RUN_TARGET_OUTPUTS file1 [file2 ...]]]
      [TEST_COMMANDS cmd1 [cmd2 ...] [TEST_OUTPUTS file1 [file2 ...]] [WORKING_DIRECTORY dir]]
      [SCRIPT_VARIABLES var1 [var2 ...]] [CTEST_TEMPLATE template])

  Function parameters:

  ``name``
    name of the test to add
  ``BUILD_TARGETS``
    videantis build targets to use for the test
  ``PRE_BUILD_TARGET_COMMANDS``
    commands to execute before the build target is executed
  ``POST_BUILD_TARGET_COMMANDS``
    commands to execute after the build target is executed
  ``ADDITIONAL_BUILD_TARGET_OUTPUTS``
    additional output files of the build target and commands
  ``RUN_TARGET``
    videantis run target to use for the test
  ``PRE_RUN_TARGET_COMMANDS``
    commands to execute before the run target is executed
  ``POST_RUN_TARGET_COMMANDS``
    commands to execute after the run target is executed
  ``ADDITIONAL_RUN_TARGET_OUTPUTS``
    additional output files of the run target and commands
  ``TEST_COMMANDS``
    commands to execute for the test
  ``WORKING_DIRECTORY``
    working directory of the test

  .. note::

    - If WORKING_DIRECTORY is not defined, the macro will use a subfolder based on the test name as working directory

  ``SCRIPT_VARIABLES``
    variables to add to the script call
  ``CTEST_TEMPLATE``
    path to a ctest template

  .. note::

    - If CTEST_TEMPLATE is not defined, the macro will use the default template shipped with the cmake module

  Generic variables available for replacement in ctest script:

  ``VID_TEST_NAME``
    name of the test
  ``VID_TEST_FILENAME``
    filename of the test script
  ``VID_TEST_BUILD_TARGET``
    boolean to check if a build target is present
  ``VID_TEST_BUILD_TARGET_COMMANDS``
    build target commands
  ``VID_TEST_BUILD_TARGET_WORKING_DIR``
    full path to build target working directory
  ``VID_TEST_PRE_BUILD_TARGET``
    boolean to check if pre build target commands are present
  ``VID_TEST_PRE_BUILD_TARGET_COMMANDS``
    pre build target commands
  ``VID_TEST_POST_BUILD_TARGET``
    boolean to check if post build target commands are present
  ``VID_TEST_POST_BUILD_TARGET_COMMANDS``
    post build target commands
  ``VID_TEST_RUN_TARGET``
    boolean to check if a run target is present
  ``VID_TEST_RUN_TARGET_COMMANDS``
    run target commands
  ``VID_TEST_RUN_TARGET_WORKING_DIR``
    full path to run target working directory
  ``VID_TEST_PRE_RUN_TARGET``
    boolean to check if pre run target commands are present
  ``VID_TEST_PRE_RUN_TARGET_COMMANDS``
    pre run target commands
  ``VID_TEST_POST_RUN_TARGET``
    boolean to check if post run target commands are present
  ``VID_TEST_POST_RUN_TARGET_COMMANDS``
    post run target commands
  ``VID_TEST_TEST``
    boolean to check if test commands are present
  ``VID_TEST_TEST_COMMANDS``
    test commands
  ``VID_TEST_WORKING_DIR``
    full path to test working directory
#]=======================================================================]
macro(vid_add_test _name)
  set(_pre_post_commands "PRE_BUILD_TARGET_COMMANDS" "POST_BUILD_TARGET_COMMANDS"
    "PRE_RUN_TARGET_COMMANDS" "POST_RUN_TARGET_COMMANDS")
  set(_outputs "TEST_OUTPUTS" "ADDITIONAL_BUILD_TARGET_OUTPUTS" "ADDITIONAL_RUN_TARGET_OUTPUTS")

  cmake_parse_arguments(_ARG "" "RUN_TARGET;CTEST_TEMPLATE;WORKING_DIRECTORY"
    "${_pre_post_commands};BUILD_TARGETS;TEST_COMMANDS;SCRIPT_VARIABLES;${_outputs}" ${ARGN})

  # check if one of the required arguments is defined
  if((NOT DEFINED _ARG_BUILD_TARGETS) AND (NOT DEFINED _ARG_RUN_TARGET) AND (NOT DEFINED _ARG_TEST_COMMANDS))
    message(FATAL_ERROR "vid_add_test() needs one of the following arguments defined to create a valid test: BUILD_TARGETS, RUN_TARGET OR TEST_COMMANDS")
  endif()

  # set default ctest template
  set(_ctest_template ${VIDSDK_CMAKE_MODULE_DIR}/templates/ctest.cmake.in)
  # check if a ctest template is provided
  if(DEFINED _ARG_CTEST_TEMPLATE)
    # check if ctest template has an absolute path
    if(IS_ABSOLUTE ${_ARG_CTEST_TEMPLATE})
      set(_ctest_template ${_ARG_CTEST_TEMPLATE})
    # if no absolute path provided assume ctest template in current source directory
    else()
      set(_ctest_template ${CMAKE_CURRENT_SOURCE_DIR}/${_ARG_CTEST_TEMPLATE})
    endif()
    # normalize path of ctest template
    cmake_path(NORMAL_PATH _ctest_template)
  endif()

  # check if ctest template can be found
  if(NOT EXISTS ${_ctest_template})
    message(FATAL_ERROR "Ctest template file not found: ${_ctest_template}")
  endif()

  # set default working directory (absolute path of CMAKE_CURRENT_BINARY_DIR)
  set(_working_dir ${CMAKE_CURRENT_BINARY_DIR}/${_name})
  # if a parameter WORKING_DIRECTORY is specified
  if(DEFINED _ARG_WORKING_DIRECTORY)
    # normalize provided path (see: https://cmake.org/cmake/help/latest/command/cmake_path.html#normalization)
    cmake_path(NORMAL_PATH _ARG_WORKING_DIRECTORY OUTPUT_VARIABLE _working_dir)
    # remove slash if last character of the path is a slash and more characters then a slash are in the string
    string(REGEX REPLACE "(.+)/$" "\\1" _working_dir ${_working_dir})

    # if the specified working directory has an absolute path
    if(IS_ABSOLUTE ${_working_dir})
      # check if specified working directory is part of the build directory
      cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${_working_dir} _wd_in_bd)
      if(NOT ${_wd_in_bd})
        message(FATAL_ERROR "WORKING_DIRECTORY has to be a directory inside the build directory!")
      endif()
    else()
      # make working directory absolute to current binary directory
      set(_working_dir ${CMAKE_CURRENT_BINARY_DIR}/${_working_dir})
    endif()
  endif()

  # if vid_add_test() was never called, create a dummy target to append files to clean
  if(NOT TARGET ctest_clean_dummy)
    add_custom_target(ctest_clean_dummy
      COMMENT "Dummy target to append created test artifacts to clean target")
  endif()

  # set defaults for build target
  set(_has_build_target FALSE)
  unset(_vid_build_targets)
  set(VID_TEST_BUILD_TARGET OFF)
  unset(VID_TEST_BUILD_TARGET_COMMANDS)
  unset(VID_TEST_BUILD_TARGET_WORKING_DIR)
  # check if a build target is defined
  foreach(_build_target ${_ARG_BUILD_TARGETS})
    # create upper case letter string from defined build target
    string(TOUPPER ${_build_target} _build_target_upper)
    # check if variable for this target is found (ensures macro are called in the correct scope and build target id defined)
    if(NOT DEFINED VID_TARGET_${_build_target_upper})
      message(FATAL_ERROR "Variable VID_TARGET_${_build_target_upper} not found. vid_add_test() might be called in the wrong scope "
        "or no videantis build target named ${_build_target} exists!")
    endif()

    # signal build target is set and found
    set(_has_build_target TRUE)
    # enable build target
    set(VID_TEST_BUILD_TARGET ON)
    # convert commands list into whitespace separated string
    string(REPLACE ";" " " _build_target_commands "${VID_TARGET_${_build_target_upper}_COMMANDS}")
    # check if build target commands are defined
    if(DEFINED VID_TEST_BUILD_TARGET_COMMANDS)
      # append new commands for the build target to list
      list(APPEND VID_TEST_BUILD_TARGET_COMMANDS && ${_build_target_commands})
    else()
      # initialize list with build target commands
      set(VID_TEST_BUILD_TARGET_COMMANDS ${_build_target_commands})
    endif()
    # check if a build target working directory is defined
    if(DEFINED VID_TEST_BUILD_TARGET_WORKING_DIR)
      # check if previously defined build target working directory is equal to working directory of this build target
      if(NOT "${VID_TEST_BUILD_TARGET_WORKING_DIR}" STREQUAL "${VID_TARGET_${_build_target_upper}_WORKING_DIR}")
        message(FATAL_ERROR "Build targets added to a test via vid_add_test() need to have the same working directory!")
      endif()
    else() # no build target working directory defined
      # set build target working directory
      set(VID_TEST_BUILD_TARGET_WORKING_DIR ${VID_TARGET_${_build_target_upper}_WORKING_DIR})
    endif()

    # check if no cmake target for the build target exists
    if(NOT TARGET build_${_build_target})
      # copy list with build target outputs
      set(_build_outputs ${VID_TARGET_${_build_target_upper}_OUTPUTS})
      # make path absolute of build target outputs
      list(TRANSFORM _build_outputs PREPEND ${CMAKE_CURRENT_BINARY_DIR}/)
      # append build target outputs to clean target
      set_property(TARGET ctest_clean_dummy APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        ${_build_outputs})
    endif()

    # append build target to list of videantis build targets used in this test
    list(APPEND _vid_build_targets VID_TARGET_${_build_target_upper})
  endforeach()

  # set defaults for pre build target
  set(VID_TEST_PRE_BUILD_TARGET OFF)
  unset(VID_TEST_PRE_BUILD_TARGET_COMMANDS)
  # check if pre build target commands are defined
  if(DEFINED _ARG_PRE_BUILD_TARGET_COMMANDS)
    # check if a build target is set and found
    if(NOT _has_build_target)
      message(FATAL_ERROR "Pre build target commands can only be defined, if a valid build target is specified!")
    endif()

    # enable pre build target
    set(VID_TEST_PRE_BUILD_TARGET ON)
    # convert commands list into whitespace separated string
    string(REPLACE ";" " " VID_TEST_PRE_BUILD_TARGET_COMMANDS "${_ARG_PRE_BUILD_TARGET_COMMANDS}")
  endif()

  # set defaults for post build target
  set(VID_TEST_POST_BUILD_TARGET OFF)
  unset(VID_TEST_POST_BUILD_TARGET_COMMANDS)
  # check if post build target commands are defined
  if(DEFINED _ARG_POST_BUILD_TARGET_COMMANDS)
    # check if a build target is set and found
    if(NOT _has_build_target)
      message(FATAL_ERROR "Post build target commands can only be defined, if a valid build target is specified!")
    endif()

    # enable post build target
    set(VID_TEST_POST_BUILD_TARGET ON)
    # convert commands list into whitespace separated string
    string(REPLACE ";" " " VID_TEST_POST_BUILD_TARGET_COMMANDS "${_ARG_POST_BUILD_TARGET_COMMANDS}")
  endif()

  # if a build target is defined and found and additional build target outputs are defined
  if(_has_build_target AND (DEFINED _ARG_ADDITIONAL_BUILD_TARGET_OUTPUTS))
    unset(_additional_build_target_outputs)
    # iterate over additional defined build target outputs
    foreach(_output ${_ARG_ADDITIONAL_BUILD_TARGET_OUTPUTS})
      # check if an absolute path is defined
      if(IS_ABSOLUTE ${_output})
        # append output with absolute path to outputs list
        list(APPEND _additional_build_target_outputs ${_output})
      else()
        # append output with absolute path of working directory to outputs list
        list(APPEND _additional_build_target_outputs ${VID_TEST_BUILD_TARGET_WORKING_DIR}/${_output})
      endif()
    endforeach()

    # append additional build target outputs to clean target
    set_property(TARGET ctest_clean_dummy APPEND PROPERTY ADDITIONAL_CLEAN_FILES
      ${_additional_build_target_outputs})
  endif()

  # set defaults for run target
  set(_has_run_target FALSE)
  set(VID_TEST_RUN_TARGET OFF)
  unset(VID_TEST_RUN_TARGET_COMMANDS)
  unset(VID_TEST_RUN_TARGET_WORKING_DIR)
  # check if a run target is defined
  if(DEFINED _ARG_RUN_TARGET)
    # create upper case letter string from defined run target
    string(TOUPPER ${_ARG_RUN_TARGET} _run_target_upper)
    # check if variable for this target is found (ensures macro are called in the correct scope and run target id defined)
    if(NOT DEFINED VID_RUN_TARGET_${_run_target_upper})
      message(FATAL_ERROR "Variable VID_RUN_TARGET_${_run_target_upper} not found. vid_add_test() might be called in the wrong scope "
        "or no videantis run target named ${_ARG_RUN_TARGET} exists!")
    endif()

    # signal run target is set and found
    set(_has_run_target TRUE)
    # enable run target
    set(VID_TEST_RUN_TARGET ON)
    # copy run target commands list
    set(VID_TEST_RUN_TARGET_COMMANDS ${VID_RUN_TARGET_${_run_target_upper}_COMMANDS})
    # find -host_app parameter
    list(FIND VID_TEST_RUN_TARGET_COMMANDS -host_app _host_app_idx)
    # check if -host_app parameter was present in run target commands list
    if(NOT ${_host_app_idx} EQUAL "-1")
      # calculate index of arguments of -host_app parameter
      math(EXPR _host_app_args_idx "${_host_app_idx} + 1")
      # get arguments of -host_app parameter
      list(GET VID_TEST_RUN_TARGET_COMMANDS ${_host_app_args_idx} _host_app_args)
      # remove arguments of -host_app parameter from list
      list(REMOVE_AT VID_TEST_RUN_TARGET_COMMANDS ${_host_app_args_idx})
      # add escaped double quotes around arguments of -host_app parameter and add it back to list
      list(INSERT VID_TEST_RUN_TARGET_COMMANDS ${_host_app_args_idx} "\\\"${_host_app_args}\\\"")
    endif()
    # convert commands list into whitespace separated string
    string(REPLACE ";" " " VID_TEST_RUN_TARGET_COMMANDS "${VID_TEST_RUN_TARGET_COMMANDS}")
    # set run target working directory
    set(VID_TEST_RUN_TARGET_WORKING_DIR ${VID_RUN_TARGET_${_run_target_upper}_WORKING_DIR})

    # check if no cmake target for the run target exists
    if(NOT TARGET run_${_ARG_RUN_TARGET})
      # copy list with run target outputs
      set(_run_outputs ${VID_RUN_TARGET_${_run_target_upper}_OUTPUTS})
      # make path absolute of run target outputs
      list(TRANSFORM _run_outputs PREPEND ${CMAKE_CURRENT_BINARY_DIR}/)
      # append run target outputs to clean target
      set_property(TARGET ctest_clean_dummy APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        ${_run_outputs})
    endif()
  endif()

  # set defaults for pre run target
  set(VID_TEST_PRE_RUN_TARGET OFF)
  unset(VID_TEST_PRE_RUN_TARGET_COMMANDS)
  # check if pre run target commands are defined
  if(DEFINED _ARG_PRE_RUN_TARGET_COMMANDS)
    set(VID_TEST_PRE_RUN_TARGET ON)
    # convert commands list into whitespace separated string
    string(REPLACE ";" " " VID_TEST_PRE_RUN_TARGET_COMMANDS "${_ARG_PRE_RUN_TARGET_COMMANDS}")
  endif()

  # set defaults for post run target
  set(VID_TEST_POST_RUN_TARGET OFF)
  unset(VID_TEST_POST_RUN_TARGET_COMMANDS)
  # check if post run target commands are defined
  if(DEFINED _ARG_POST_RUN_TARGET_COMMANDS)
    set(VID_TEST_POST_RUN_TARGET ON)
    # convert commands list into whitespace separated string
    string(REPLACE ";" " " VID_TEST_POST_RUN_TARGET_COMMANDS "${_ARG_POST_RUN_TARGET_COMMANDS}")
  endif()

  # if a run target is det and found and additional run target outputs are defined
  if(_has_run_target AND (DEFINED _ARG_ADDITIONAL_RUN_TARGET_OUTPUTS))
    unset(_additional_run_target_outputs)
    # iterate over additional defined run target outputs
    foreach(_output ${_ARG_ADDITIONAL_RUN_TARGET_OUTPUTS})
      # check if an absolute path is defined
      if(IS_ABSOLUTE ${_output})
        # append output with absolute path to outputs list
        list(APPEND _additional_run_target_outputs ${_output})
      else()
        # append output with absolute path of working directory to outputs list
        list(APPEND _additional_run_target_outputs ${VID_RUN_TARGET_${_run_target_upper}_WORKING_DIR}/${_output})
      endif()
    endforeach()

    # append additional run target outputs to clean target
    set_property(TARGET ctest_clean_dummy APPEND PROPERTY ADDITIONAL_CLEAN_FILES
      ${_additional_run_target_outputs})
  endif()

  # set defaults for test
  set(VID_TEST_TEST OFF)
  unset(VID_TEST_TEST_COMMANDS)
  # check if test commands are defined
  if(DEFINED _ARG_TEST_COMMANDS)
    set(VID_TEST_TEST ON)
    # convert commands list into whitespace separated string
    string(REPLACE ";" " " VID_TEST_TEST_COMMANDS "${_ARG_TEST_COMMANDS}")
  endif()

  # check if test outputs are defined
  if(DEFINED _ARG_TEST_OUTPUTS)
    unset(_test_outputs)
    # iterate over defined test outputs
    foreach(_output ${_ARG_TEST_OUTPUTS})
      # check if an absolute path is defined
      if(IS_ABSOLUTE ${_output})
        # append output with absolute path to outputs list
        list(APPEND _test_outputs ${_output})
      else()
        # append output with absolute path of working directory to outputs list
        list(APPEND _test_outputs ${_working_dir}/${_output})
      endif()
    endforeach()

    # append test outputs to clean target
    set_property(TARGET ctest_clean_dummy APPEND PROPERTY ADDITIONAL_CLEAN_FILES
      ${_test_outputs})
  endif()

  unset(_script_variables)
  # check if script variables are defined
  if(DEFINED _ARG_SCRIPT_VARIABLES)
    # append each script variable with -D to a list
    foreach(_script_var ${_ARG_SCRIPT_VARIABLES})
      list(APPEND _script_variables -D ${_script_var})
    endforeach(_script_var)
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_test() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # set output ctest script filename
  set(_ctest_script ${_name}.cmake)

  # set generic variables
  set(VID_TEST_NAME ${_name})
  set(VID_TEST_FILENAME ${_ctest_script})
  set(VID_TEST_WORKING_DIR ${_working_dir})

  # configure and generate (generate replace generator expressions) ctest script
  # read template, using a cache to avoid re-reading the same file for every test
  string(MAKE_C_IDENTIFIER "${_ctest_template}" _ctest_template_cache_key)
  get_property(_ctest_template_cached GLOBAL PROPERTY _VID_CTEST_TEMPLATE_CACHE_${_ctest_template_cache_key} SET)
  if(_ctest_template_cached)
    get_property(_ctest_template_string GLOBAL PROPERTY _VID_CTEST_TEMPLATE_CACHE_${_ctest_template_cache_key})
  else()
    file(READ ${_ctest_template} _ctest_template_string)
    set_property(GLOBAL PROPERTY _VID_CTEST_TEMPLATE_CACHE_${_ctest_template_cache_key} "${_ctest_template_string}")
  endif()
  unset(_ctest_template_cache_key)
  unset(_ctest_template_cached)
  # configure content read from file (like configure_file only on a string)
  string(CONFIGURE ${_ctest_template_string} _ctest_template_string @ONLY)
  # generate output file
  file(GENERATE OUTPUT ${_working_dir}/${_ctest_script} CONTENT ${_ctest_template_string})

  # define test based on working directory and with the generated cmake script to execute test
  add_test(NAME ${_name}
    COMMAND ${CMAKE_COMMAND} ${_script_variables} -P ${_ctest_script}
    WORKING_DIRECTORY ${_working_dir})

  # if the test is using one or more build targets, create a resource lock to avoid parallel execution of the build targets
  if(_has_build_target)
    set_property(TEST ${_name} APPEND PROPERTY RESOURCE_LOCK ${_vid_build_targets})
  endif()
  # if the test is using a run target, create a resource lock to avoid parallel execution of the run target
  if(_has_run_target)
    set_property(TEST ${_name} APPEND PROPERTY RESOURCE_LOCK VID_RUN_TARGET_${_run_target_upper})
  endif()
endmacro(vid_add_test)
