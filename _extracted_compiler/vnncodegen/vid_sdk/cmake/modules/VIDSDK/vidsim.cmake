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
# FILENAME:    vidsim.cmake
#
# DESCRIPTION: Support file for videantis SDK simulator
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

#[=======================================================================[.rst:
.. cmake:command:: vid_test_vidsim_arg

  Macro to test if an argument is known by the videantis simulator.
  This macro is intended for internal use.

  .. code-block:: cmake

    vid_test_vidsim_arg(argument [TOOL tool] [SET_CACHE])

  Function parameters:

  ``argument``
    videantis simulator parameter to test
  ``TOOL``
    define vidsim binary to test (vidsim, vidsim-cct and vidsim-man; default: vidsim)
  ``SET_CACHE``
    if set, create cmake cache entry <TOOL>_HAS_<ARG>
  ``OUTPUT``
    _VIDSIM_ARG
#]=======================================================================]
macro(vid_test_vidsim_arg _argument)
  cmake_parse_arguments(_ARG "SET_CACHE" "TOOL"
    "" ${ARGN})

  # define vidsim as default tool
  set(_tool VIDSIM)
  # check if TOOL is specified
  if(DEFINED _ARG_TOOL)
    # overwrite default tool variable
    string(TOUPPER ${_ARG_TOOL} _tool)
  endif()

  # set found videantis simulator argument to true by default
  set(_VIDSIM_ARG TRUE)

  # run vidsim with argument to test
  execute_process(COMMAND ${${_tool}_EXECUTABLE} ${_argument}
    OUTPUT_VARIABLE _vidsim_output
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # check if vidsim returns "unknown argument"
  if(${_vidsim_output} MATCHES "unknown argument '${_argument}'")
    set(_VIDSIM_ARG FALSE)
  endif()

  # check if result should be written into cache
  if(${_ARG_SET_CACHE})
    # remove hyphen from argument and make it upper case
    string(REPLACE "-" "" _argument_name ${_argument})
    string(TOUPPER ${_argument_name} _argument_name_upper)

    # set variable in cache with test result
    set(${_tool}_HAS_${_argument_name_upper} ${_VIDSIM_ARG}
      CACHE BOOL "${_${_variant}_prettified_name} supports argument ${_argument}")
  endif()
endmacro(vid_test_vidsim_arg)

# define vidsim variants to test for arguments
set(_vidsim_variants vidsim vidsim-cct vidsim-man vidsim-dbg)

# define latest simulator arguments that generate output files
set(_vidsim_output_args -log -dh -host_app_stdout -tf -tl -th -outfile
  -stdoutfile -prof -prof_ext -prof_instr -prof_cyc -prof_dma -prof_mem
  -prof_summary -prof_summary_sub -di -dd)
# define latest code coverage arguments that generate output files (including vidsim arguments)
set(_vidsim-cct_output_args ${_vidsim_output_args} -cct_report -cct_data)
# define latest memory analyzer arguments that generate output files (including vidsim arguments)
set(_vidsim-man_output_args ${_vidsim_output_args} -man)
# define latest debugger arguments that generate output files (including vidsim arguments)
set(_vidsim-dbg_output_args ${_vidsim_output_args})

# define latest simulator arguments that require input files
set(_vidsim_input_args -lh -li -ld -map)
# define latest code coverage arguments that require input files (including vidsim arguments)
set(_vidsim-cct_input_args ${_vidsim_input_args} -cct_user_html)
# define latest memory analyzer arguments that require input files (including vidsim arguments)
set(_vidsim-man_input_args ${_vidsim_input_args})
# define latest debugger arguments that require input files (including vidsim arguments)
set(_vidsim-dbg_input_args ${_vidsim_input_args})

# set a prettified name for vidsim
set(_vidsim_prettified_name "videantis simulator")
# set a prettified name for vidsim-cct
set(_vidsim-cct_prettified_name "videantis simulator code coverage extension")
# set a prettified name for vidsim-man
set(_vidsim-man_prettified_name "videantis simulator memory analyzer extension")
# set a prettified name for vidsim-dbg
set(_vidsim-dbg_prettified_name "videantis simulator GDB/MI version")

# iterate over all vidsim variants
foreach(_variant ${_vidsim_variants})
  # get a all upper case version of the current variant
  string(TOUPPER ${_variant} _variant_upper)

  message(STATUS "Testing for ${_variant} features")

  # test for argument "-log" and set VIDSIM_HAS_LOG cached variable
  vid_test_vidsim_arg(-log TOOL ${_variant} SET_CACHE)
  message(STATUS "Testing for ${_variant} features - ${_variant_upper}_HAS_LOG: ${${_variant_upper}_HAS_LOG}")
  # test for argument "-host_app" and set VIDSIM_HAS_HOST_APP cached variable
  vid_test_vidsim_arg(-host_app TOOL ${_variant} SET_CACHE)
  message(STATUS "Testing for ${_variant} features - ${_variant_upper}_HAS_HOST_APP: ${${_variant_upper}_HAS_HOST_APP}")

  # iterate over latest vidsim arguments that generate output files
  foreach(_argument ${_${_variant}_output_args})
    # test for current argument from list
    # output: _VIDSIM_ARG
    vid_test_vidsim_arg(${_argument} TOOL ${_variant})

    # if videantis simulator knows the argument add to list of found arguments
    if(${_VIDSIM_ARG})
      list(APPEND _${_variant}_output_args_found ${_argument})
    endif()
  endforeach(_argument)

  # define found simulator arguments as cached global variable
  set(${_variant_upper}_OUTPUT_ARGS ${_${_variant}_output_args_found}
    CACHE STRING "${_${_variant}_prettified_name} arguments that generate output files")

  # iterate over latest vidsim arguments that require input files
  foreach(_argument ${_${_variant}_input_args})
    # test for current argument from list
    # output: _VIDSIM_ARG
    vid_test_vidsim_arg(${_argument} TOOL ${_variant})

    # if videantis simulator knows the argument add to list of found arguments
    if(${_VIDSIM_ARG})
      list(APPEND _${_variant}_input_args_found ${_argument})
    endif()
  endforeach(_argument)

  # define found simulator arguments as cached global variable
  set(${_variant_upper}_INPUT_ARGS ${_${_variant}_input_args_found}
    CACHE STRING "${_${_variant}_prettified_name} arguments that require input files")
endforeach(_variant)
