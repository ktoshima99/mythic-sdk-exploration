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
# FILENAME:    run.cmake
#
# DESCRIPTION: Support file for videantis SDK to run/simulate
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

#[=======================================================================[.rst:
.. cmake:command:: _get_num_of_cores

  Macro to get the number of cores from a simulator/debugger configuration
  template. This macro is intended for internal use within the macros
  :cmake:command:`vid_init_run_target` and :cmake:command:`vid_debug_run_target`.
  Don’t call this macro directly!

  .. code-block:: cmake

    _get_num_of_cores(target template context)

  Function parameters:

  ``target``
    name of the target
  ``template``
    simulator or debugger configuration template
  ``context``
    set context (debug or run)

  Exposed variables:

  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_NUM_MPS``
    number of v-MPs for the run target parsed from simcfg/dbgcfg template
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_NUM_SPS``
    number of v-SPs for the run target parsed from simcfg/dbgcfg template (variable only exposed, if v-SP tool chain is available)

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
#]=======================================================================]
macro(_get_num_of_cores _target _template _context)
  # get target in all upper case letters
  string(TOUPPER ${_target} _target_upper)

  # check if simulator/debugger configuration template file exists
  if(NOT EXISTS ${_template})
    message(FATAL_ERROR "Simulator/Debugger configuration template file ${_template} could not found")
  endif()

  # set allowed contexts
  set(_ALLOWED_CONTEXTS run debug)
  # check context for correctness
  if(NOT ${_context} IN_LIST _ALLOWED_CONTEXTS)
    message(FATAL_ERROR "Macro _get_num_of_cores() is called in an invalid context: ${_context}")
  endif()

  # set default generic prefix for generated variables
  set(_generic_prefix VID_RUN_TARGET)
  # check if provided context is debug
  if(${_context} STREQUAL "debug")
    # overwrite generic prefix for debug context
    set(_generic_prefix VID_DEBUG_RUN_TARGET)
  endif()

  # parse simcfg template as cmake list
  # output: _SIMCFG_LIST
  vid_parse_simcfg_as_list(FROM_FILE ${_template})

  # get number of v-MP cores from simcfg
  # output: _SIMCFG_ARG
  vid_get_arg_from_simcfg_list(-cores AFTER ^mp SIMCFG ${_SIMCFG_LIST})
  # check if v-MP cores are found
  if(DEFINED _SIMCFG_ARG)
    # save the number of found v-MP cores to variable
    set(${_generic_prefix}_${_target_upper}_NUM_MPS ${_SIMCFG_ARG})
  endif()

  # get number of v-MP cores from simcfg when cpu type should be inserted
  # output: _SIMCFG_ARG
  vid_get_arg_from_simcfg_list(-cores AFTER @${_generic_prefix}_CPU_MP@ SIMCFG ${_SIMCFG_LIST})
  # check if v-MP cores are found
  if(DEFINED _SIMCFG_ARG)
    # check if v-MP cores are already found
    if(DEFINED ${_generic_prefix}_${_target_upper}_NUM_MPS)
      message(FATAL_ERROR "Multiple definitions of v-MP cores are found in simulator configuration template, please check!")
    else()
      # save the number of found v-MP cores to variable
      set(${_generic_prefix}_${_target_upper}_NUM_MPS ${_SIMCFG_ARG})
    endif()
  endif()

  # if no v-MP cores are found in simulator configuration template
  if(NOT DEFINED ${_generic_prefix}_${_target_upper}_NUM_MPS)
    # set the number of v-MPs to zero
    set(${_generic_prefix}_${_target_upper}_NUM_MPS 0)
  endif()

  # check if v-SP tool chain is available
  if(${VIDSDK_HAS_VSP_TOOLCHAIN})
    # get number of v-SP cores from simcfg
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(-cores AFTER ^sp SIMCFG ${_SIMCFG_LIST})
    # check if v-SP cores are found
    if(DEFINED _SIMCFG_ARG)
      # save the number of found v-SP cores to variable
      set(${_generic_prefix}_${_target_upper}_NUM_SPS ${_SIMCFG_ARG})
    endif()

    # get number of v-SP cores from simcfg when cpu type should be inserted
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(-cores AFTER @${_generic_prefix}_CPU_SP@ SIMCFG ${_SIMCFG_LIST})
    # check if v-SP cores are found
    if(DEFINED _SIMCFG_ARG)
      # check if v-SP cores are already found
      if(DEFINED ${_generic_prefix}_${_target_upper}_NUM_SPS)
        message(FATAL_ERROR "Multiple definitions of v-SP cores are found in simulator configuration template, please check!")
      else()
        # save the number of found v-SP cores to variable
        set(${_generic_prefix}_${_target_upper}_NUM_SPS ${_SIMCFG_ARG})
      endif()
    endif()

    # if no v-SP cores are found in simulator configuration template
    if(NOT DEFINED ${_generic_prefix}_${_target_upper}_NUM_SPS)
      # set the number of v-SPs to zero
      set(${_generic_prefix}_${_target_upper}_NUM_SPS 0)
    endif()

    # check if number of v-MPs and v-SPs parsed from simcfg is greater then 0
    # (implicitly checks if result is a number)
    # TODO: define a maximum number of v-MPs and v-SPs to check against
    if((NOT ${${_generic_prefix}_${_target_upper}_NUM_MPS} GREATER 0) AND (NOT ${${_generic_prefix}_${_target_upper}_NUM_SPS} GREATER 0))
      message(FATAL_ERROR "Found invalid number of v-MPs and v-SPs in simulator configuration template: ${_template}")
    endif()
  else()
    # check if number of v-MPs parsed from simcfg is greater then 0
    # (implicitly checks if result is a number)
    # TODO: define a maximum number of v-MPs to check against
    if(NOT ${${_generic_prefix}_${_target_upper}_NUM_MPS} GREATER 0)
      message(FATAL_ERROR "Found invalid number of v-MPs in simulator configuration template: ${_template}")
    endif()
  endif()
endmacro(_get_num_of_cores)

#[=======================================================================[.rst:
.. cmake:command:: vid_init_run_target

  Macro to initialize a run/debugging of a videantis build target.
  Run target gets defined and some default values get initialized.

  .. code-block:: cmake

    vid_init_run_target(name [SIMCFG_TEMPLATE template]
        [DBGCFG_TEMPLATE template] [DEBUG_ONLY])

  Function parameters:

  ``name``
    name for the run target
  ``SIMCFG_TEMPLATE template``
    path to a simulator configuration template

  .. note::

    - If SIMCFG_TEMPLATE is not defined, the macro will search for a simcfg template named <target_name>.simcfg.in
    - If DEBUG_ONLY is set, SIMCFG_TEMPLATE will be ignored

  ``DBGCFG_TEMPLATE template``
    path to a debugger configuration template

  .. note::

    - If DBGCFG_TEMPLATE is not defined, the macro will search for a dbgcfg template named <target_name>.dbgcfg.in when DEBUG_ONLY is set

  ``DEBUG_ONLY``
    initialize a debugger only run target

  Exposed variables:

  ``VID_RUN_TARGETS_<SOURCE_DIR_IDENTIFIER>``
    list of run target defined in current scope
  ``VID_RUN_TARGET_<TARGET_NAME>``
    name of run target (case sensitive)
  ``VID_RUN_TARGET_<TARGET_NAME>_DEBUG_ONLY``
    run target is debugger only
  ``VID_RUN_TARGET_<TARGET_NAME>_SIMCFG_TEMPLATE``
    full path to the simulator configuration template
  ``VID_DEBUG_RUN_TARGET_<TARGET_NAME>_DBGCFG_TEMPLATE``
    full path to the debugger configuration template
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_NUM_MPS``
    number of v-MPs for the run target parsed from simcfg/dbgcfg template
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_NUM_SPS``
    number of v-SPs for the run target parsed from simcfg/dbgcfg template (variable only exposed, if v-SP tool chain is available)
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

  ``<SOURCE_DIR_IDENTIFIER>``
    current source directory name in all upper case letters
  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<target_name>``
    target name
  ``<ID>``
    ID of a target core
#]=======================================================================]
macro(vid_init_run_target _name)
  cmake_parse_arguments(_ARG "DEBUG_ONLY" "SIMCFG_TEMPLATE;DBGCFG_TEMPLATE"
    "" ${ARGN})

  # set name with upper case
  string(TOUPPER ${_name} _name_upper)

  # if we are not in the root directory of the cmake project
  if(NOT ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    # create an identifier based on the relative path of current source dir to project root
    string(REPLACE ${CMAKE_SOURCE_DIR}/ "" _current_source_dir_relative ${CMAKE_CURRENT_SOURCE_DIR})
    string(REPLACE "/" "_" _source_dir_identifier ${_current_source_dir_relative})
    string(TOUPPER ${_source_dir_identifier} _source_dir_identifier)
  else()
    # in the project root set root as identifier
    set(_source_dir_identifier ROOT)
  endif()

  # check if run target is already defined for the actual binary directory
  if(${_name_upper} IN_LIST VID_RUN_TARGETS_${_source_dir_identifier})
    message(FATAL_ERROR "${_name} (case insensitive) is already defined as run target in scope of ${CMAKE_CURRENT_BINARY_DIR}")
  else()
    # append run target to list of run targets of current source directory
    list(APPEND VID_RUN_TARGETS_${_source_dir_identifier} ${_name_upper})
  endif()

  # define videantis run target based on name
  set(VID_RUN_TARGET_${_name_upper} ${_name})
  # save debug only setting into a target related variable
  set(VID_RUN_TARGET_${_name_upper}_DEBUG_ONLY ${_ARG_DEBUG_ONLY})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_init_run_target() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # check if this a debug only run target and a simulator configuration is specified
  if(${_ARG_DEBUG_ONLY} AND (DEFINED _ARG_SIMCFG_TEMPLATE))
    # raise a warning that the simcfg template will be ignored
    message(WARNING "A simulator configuration template is specified for a debugging only run target.
      The template ${_ARG_SIMCFG_TEMPLATE} will be ignored")
  endif()

  # prepare with simcfg template only in not debug only mode
  if(NOT ${_ARG_DEBUG_ONLY})
    # if SIMCFG_TEMPLATE is defined as argument
    if(DEFINED _ARG_SIMCFG_TEMPLATE)
      # set SIMCFG_TEMPLATE as simulator configuration template file
      set(VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE ${_ARG_SIMCFG_TEMPLATE})
    else()
      # set default simulator configuration template file
      set(VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE ${_name}.simcfg.in)
      message(WARNING "If SIMCFG_TEMPLATE is not defined with vid_init_run_target(), "
        "a default simcfg template will be assumed: ${VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE}")
    endif()

    # if simcfg_template is not a absolute path
    if(NOT IS_ABSOLUTE ${VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE})
      # create absolute path from relative path of simcfg_template
      get_filename_component(VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE ${VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE} ABSOLUTE)
    endif()

    # check if simulator configuration template file exists
    if(NOT EXISTS ${VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE})
      message(FATAL_ERROR "Simulator configuration template file ${VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE} could not found")
    endif()

    # get number of cores from simcfg template for run context
    _get_num_of_cores(${_name} ${VID_RUN_TARGET_${_name_upper}_SIMCFG_TEMPLATE} run)

    # if DBGCFG_TEMPLATE is defined as additional argument
    if(DEFINED _ARG_DBGCFG_TEMPLATE)
      # set DBGCFG_TEMPLATE as debugger configuration template file
      set(VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE ${_ARG_DBGCFG_TEMPLATE})

      # if dbgcfg_template is not a absolute path
      if(NOT IS_ABSOLUTE ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE})
        # create absolute path from relative path of dbgcfg_template
        get_filename_component(VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE} ABSOLUTE)
      endif()

      # check if debugger configuration template file exists
      if(NOT EXISTS ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE})
        message(FATAL_ERROR "Debugger configuration template file ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE} could not found")
      endif()

      # get number of cores from dbgcfg template for debug context
      _get_num_of_cores(${_name} ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE} debug)
    endif()
  else()
    # if DBGCFG_TEMPLATE is defined as argument
    if(DEFINED _ARG_DBGCFG_TEMPLATE)
      # set DBGCFG_TEMPLATE as debugger configuration template file
      set(VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE ${_ARG_DBGCFG_TEMPLATE})
    else()
      # set default debugger configuration template file
      set(VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE ${_name}.dbgcfg.in)
      message(WARNING "If DBGCFG_TEMPLATE is not defined with vid_init_run_target(), "
        "a default dbgcfg template will be assumed: ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE}")
    endif()

    # if dbgcfg_template is not a absolute path
    if(NOT IS_ABSOLUTE ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE})
      # create absolute path from relative path of dbgcfg_template
      get_filename_component(VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE} ABSOLUTE)
    endif()

    # check if debugger configuration template file exists
    if(NOT EXISTS ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE})
      message(FATAL_ERROR "Debugger configuration template file ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE} could not found")
    endif()

    # get number of cores from dbgcfg template for debug context
    _get_num_of_cores(${_name} ${VID_DEBUG_RUN_TARGET_${_name_upper}_DBGCFG_TEMPLATE} debug)
  endif()

  # generate target consts cmake variables
  vid_generate_target_consts()
endmacro(vid_init_run_target)

#[=======================================================================[.rst:
.. cmake:command:: vid_map_build_target

  Macro to map a videantis build target to v-MPs/v-SPs for simulation.
  This marcos requires a build target via BUILD and one or more v-MP core ids via MPS
  to map a build target to a run target. If v-SP tool chain is available, SPS can be used
  to map one or more v-SP core ids to from a built target to a run target.
  The macro will ensure that the v-MP/v-SP core is not mapped to another build target for
  this run target. To map more then one build target call the macro again.

  .. code-block:: cmake

    vid_map_build_target(target BUILD build_target
        [MPS mpId1 [mpId2 ...]] [SPS spId1 [spId2 ...]])

  Function parameters:

  ``target``
    name of an already defined run target
  ``BUILD``
    build target to map to the run target
  ``MPS``
    map v-MPs to run target for defined build target
  ``SPS``
    map v-SPs to run target for defined build target (option only available, if v-SP tool chain is available)

  .. note::

    If v-SP tool chain is available, MPS or SPS needs to be set (setting both is not allowed).
    For the case v-SP tool chain is not available, MPS is a mandatory option.

  Exposed variables:

  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_TARGET_CPU_MP``
    target cpu v-MP inherited from mapped build target(s)
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_TARGET_CPU_SP``
    target cpu v-SP inherited from mapped build target(s)
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_TARGET_SOC``
    target soc inherited from mapped build target(s)
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_MP_MAP``
    v-MP core IDs that are mapped to this run target
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_SP_MAP``
    v-SP core IDs that are mapped to this run target
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_BIN_FILE_MP<MP_ID>``
    full path to the bin file for v-MP with ID X
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_MAP_FILE_MP<MP_ID>``
    full path to the map file for v-MP with ID X
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_BIN_FILE_SP<SP_ID>``
    full path to the bin file for v-SP with ID X
  ``VID_(DEBUG_)RUN_TARGET_<TARGET_NAME>_MAP_FILE_SP<SP_ID>``
    full path to the map file for v-SP with ID X

  .. note::

    Depending on the build target mapped only v-MP or v-SP depending variables are exposed.

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<MP_ID>``
    ID of a v-MP core
  ``<SP_ID>``
    ID of a v-SP core
#]=======================================================================]
macro(vid_map_build_target _target)
  set(_multiValueArgs MPS)
  # check if v-SP tool chain is available
  if(${VIDSDK_HAS_VSP_TOOLCHAIN})
    list(APPEND _multiValueArgs SPS)
  endif()
  cmake_parse_arguments(_ARG "" "BUILD"
    "${_multiValueArgs}" ${ARGN})

  if(${VIDSDK_HAS_VSP_TOOLCHAIN})
    # check if argument MPS or SPS is defined
    if((NOT DEFINED _ARG_MPS) AND (NOT DEFINED _ARG_SPS))
      message(FATAL_ERROR "Argument MPS or SPS needs to be defined to map the videantis build target to one or more v-MPs/v-SPs")
    endif()
    # check if arguments MPS and SPS are both defined
    if((DEFINED _ARG_MPS) AND (DEFINED _ARG_SPS))
      message(FATAL_ERROR "Arguments MPS and SPS cannot defined at the same macro call of vid_map_build_target(). Map v-MPs and v-SPs with separate calls")
    endif()

    # check if argument BUILD is defined
    if(NOT DEFINED _ARG_BUILD)
      message(FATAL_ERROR "Argument BUILD needs to be defined to map the videantis build target to one or more v-MPs/v-SPs")
    endif()
  # no v-SP tool chain
  else()
    # check if argument MPS is defined
    if(NOT DEFINED _ARG_MPS)
      message(FATAL_ERROR "Argument MPS needs to be defined to map the videantis build target to one or more v-MPs")
    endif()

    # check if argument BUILD is defined
    if(NOT DEFINED _ARG_BUILD)
      message(FATAL_ERROR "Argument BUILD needs to be defined to map the videantis build target to one or more v-MPs")
    endif()
  endif() # end VIDSDK_HAS_VSP_TOOLCHAIN

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_build_target() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if videantis run target is defined
  if(NOT DEFINED VID_RUN_TARGET_${_target_upper})
    message(FATAL_ERROR "Run target ${_target} is not found in videantis build context")
  endif()

  # set default generic prefix for generated variables
  set(_generic_prefix VID_RUN_TARGET)
  # check if this is a debug only run target
  if(${VID_RUN_TARGET_${_target_upper}_DEBUG_ONLY})
    # set generic prefix for debug only
    set(_generic_prefix VID_DEBUG_RUN_TARGET)
  endif()

  # set build_target
  set(_build_target ${_ARG_BUILD})
  # set build_target with upper case
  string(TOUPPER ${_build_target} _build_target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_build_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_build_target})
    message(FATAL_ERROR "Build target ${_build_target} is not found in videantis build context")
  endif()

  # check if vid_build_target() was run for defined target
  # vid_build_target() defines variables for map and bin file
  get_property(_bin_file GLOBAL PROPERTY VID_TARGET_${_build_target_upper}_BIN_FILE)
  get_property(_map_file GLOBAL PROPERTY VID_TARGET_${_build_target_upper}_MAP_FILE)
  if(NOT ((DEFINED _bin_file) AND (DEFINED _map_file)))
    message(FATAL_ERROR "map and/or bin file could not be found for target ${_build_target}. "
      "vid_build_target() needs to be called to generate map and bin file.")
  endif()

  # set core type by default to v-MP
  set(_core_type MP)
  # if SPS is defined (check before that only MPS or SPS is defined)
  if(DEFINED _ARG_SPS)
    # set core type to v-SP
    set(_core_type SP)
  endif()

  # check if target type matches selected core type to map
  get_property(_target_type GLOBAL PROPERTY VID_TARGET_${_build_target_upper}_TYPE)
  if(NOT ${_target_type} STREQUAL "V${_core_type}")
    message(FATAL_ERROR "The specified target ${_build_target} is not a target for core type v-${_core_type}")
  endif()

  # get number of cores to map
  list(LENGTH _ARG_${_core_type}S _cores_to_map)
  # get number of already mapped cores of specific type
  list(LENGTH ${_generic_prefix}_${_target_upper}_${_core_type}_MAP _mapped_cores)
  # calculate number of free cores
  math(EXPR _free_cores "${${_generic_prefix}_${_target_upper}_NUM_${_core_type}S} - ${_mapped_cores}")
  # calculate the maximum core id
  math(EXPR _max_core_id "${${_generic_prefix}_${_target_upper}_NUM_${_core_type}S} - 1")
  # get a list without duplicated core ids
  set(_rm_dups_cores ${_ARG_${_core_type}S})
  list(REMOVE_DUPLICATES _rm_dups_cores)
  list(LENGTH _rm_dups_cores _num_cores_wo_dups)
  # check if a core is defined more then once
  if(NOT ${_cores_to_map} EQUAL ${_num_cores_wo_dups})
    message(FATAL_ERROR "One or more v-${_core_type}s are defined more the once in the macro call of vid_map_build_target()")
  endif()
  # check if more cores shall be mapped then free cores are available
  if(${_free_cores} LESS ${_cores_to_map})
    message(FATAL_ERROR "Not enough v-${_core_type}s are defined in simulator/debugger configuration template. "
      "Mapping the defined ${_cores_to_map} v-${_core_type}s would exceed the overall available ${${_generic_prefix}_${_target_upper}_NUM_${_core_type}S} v-${_core_type}s")
  endif()
  foreach(_core ${_ARG_${_core_type}S})
    # check if current core is already mapped
    if(${_core} IN_LIST ${_generic_prefix}_${_target_upper}_${_core_type}_MAP)
      message(FATAL_ERROR "v-${_core_type} ${_core} is already mapped to this run target ${_target} with a different videantis build target")
    endif()
    # check if current core is extending the range based on number of cores configured
    if(${_core} GREATER ${_max_core_id})
      message(FATAL_ERROR "The current v-${_core_type} ${_core} is out range for the number of v-${_core_type}s (${${_generic_prefix}_${_target_upper}_NUM_${_core_type}S}) for simulation/debugging")
    endif()

    # append current core to list of mapped core for the defined type
    list(APPEND ${_generic_prefix}_${_target_upper}_${_core_type}_MAP ${_core})
    # assign map and bin file of videantis build target to current core
    set(${_generic_prefix}_${_target_upper}_BIN_FILE_${_core_type}${_core} ${_bin_file})
    set(${_generic_prefix}_${_target_upper}_MAP_FILE_${_core_type}${_core} ${_map_file})
  endforeach(_core)

  # get target cpu from build target
  get_property(_target_cpu GLOBAL PROPERTY VID_TARGET_${_build_target_upper}_TARGET_CPU)

  # check if target cpu for build target was defined
  if(NOT DEFINED _target_cpu)
    message(FATAL_ERROR "No target cpu exposed for build target ${_build_target}")
  endif()

  # check if target cpu for core type is already defined from other build target mapping
  # else set target cpu
  if(DEFINED ${_generic_prefix}_${_target_upper}_TARGET_CPU_${_core_type})
    # if already set cpu type is not equal to build target cpu raise error
    if(NOT ${_target_cpu} STREQUAL ${_generic_prefix}_${_target_upper}_TARGET_CPU_${_core_type})
      message(FATAL_ERROR "Build target ${_build_target} cpu (${_target_cpu}) does not match with previously mapped build targets cpu (${${_generic_prefix}_${_target_upper}_TARGET_CPU_${_core_type}})")
    endif()
  else()
    set(${_generic_prefix}_${_target_upper}_TARGET_CPU_${_core_type} ${_target_cpu})
  endif()

  # get target soc from build target
  get_property(_target_soc GLOBAL PROPERTY VID_TARGET_${_build_target_upper}_TARGET_SOC)

  # check if target soc for build target was defined
  if(NOT DEFINED _target_soc)
    message(FATAL_ERROR "No target soc exposed for build target ${_build_target}")
  endif()

  # check if target soc is already defined from other build target mapping
  # else set target soc
  if(DEFINED ${_generic_prefix}_${_target_upper}_TARGET_SOC)
    # if already set soc type is not equal to build target soc raise error
    if(NOT ${_target_soc} STREQUAL ${_generic_prefix}_${_target_upper}_TARGET_SOC)
      message(FATAL_ERROR "Build target ${_build_target} soc (${_target_soc}) does not match with previously mapped build targets soc (${${_generic_prefix}_${_target_upper}_TARGET_SOC})")
    endif()
  else()
    set(${_generic_prefix}_${_target_upper}_TARGET_SOC ${_target_soc})
  endif()
endmacro(vid_map_build_target)

#[=======================================================================[.rst:
.. cmake:command:: _prepare_cfg_variables

  Macro to prepare variables to be inserted into templates of simulator/debugger
  configuration. This macro is intended for internal use within the macros
  :cmake:command:`vid_run_target` and :cmake:command:`vid_debug_run_target`.
  Don't call this macro directly!

  .. code-block:: cmake

    _prepare_cfg_variables(target context working_dir)

  Function parameters:

  ``target``
    name of an already defined run target
  ``context``
    set context (debug or run)
  ``working_dir``
    working directory of run/debug target

  Generic variables available for replacement in videantis simulator/debugger configuration:

  ``VID_(DEBUG_)RUN_TARGET``
    name of the run target
  ``VID_(DEBUG_)RUN_TARGET_SOURCE_DIR``
    full path to source dir of run target
  ``VID_(DEBUG_)RUN_TARGET_DIR``
    full path to dir of run target
  ``VID_(DEBUG_)RUN_TARGET_CPU_MP``
    target cpu v-MP
  ``VID_(DEBUG_)RUN_TARGET_CPU_SP``
    target cpu v-SP
  ``VID_(DEBUG_)RUN_TARGET_SOC``
    target soc
  ``VID_(DEBUG_)RUN_TARGET_BIN_FILE_MP<MP_ID>``
    full path to bin file for v-MP core
  ``VID_(DEBUG_)RUN_TARGET_MAP_FILE_MP<MP_ID>``
    full path to map file for v-MP core
  ``VID_(DEBUG_)RUN_TARGET_BIN_FILE_SP<SP_ID>``
    full path to bin file for v-SP core
  ``VID_(DEBUG_)RUN_TARGET_MAP_FILE_SP<SP_ID>``
    full path to map file for v-SP core
  ``VID_(DEBUG_)RUN_TARGET_SIMCFG/DBGCFG_PATH``
    full path to folder of simulator configuration template

  Placeholder:

  ``<MP_ID>``
    ID of a v-MP core
  ``<SP_ID>``
    ID of a v-SP core
#]=======================================================================]
macro(_prepare_cfg_variables _target _context _working_dir)
  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if videantis run target is defined
  if(NOT DEFINED VID_RUN_TARGET_${_target_upper})
    message(FATAL_ERROR "Run target ${_target} is not found in videantis build context")
  endif()

  # set allowed contexts
  set(_ALLOWED_CONTEXTS run debug)
  # check context for correctness
  if(NOT ${_context} IN_LIST _ALLOWED_CONTEXTS)
    message(FATAL_ERROR "Macro _prepare_cfg_variables() is called in an invalid context: ${_context}")
  endif()

  # set default generic prefix and simulator configuration type for generated variables
  set(_generic_prefix VID_RUN_TARGET)
  set(_cfg_type SIMCFG)
  # check if provided context is debug
  if(${_context} STREQUAL "debug")
    # overwrite generic prefix and simulator configuration type for debug context
    set(_generic_prefix VID_DEBUG_RUN_TARGET)
    set(_cfg_type DBGCFG)
  endif()

  # set default map prefix for getting generated
  set(_map_prefix VID_RUN_TARGET)
  # check if debug only run target
  if(${VID_RUN_TARGET_${_target_upper}_DEBUG_ONLY})
    # overwrite map prefix for debug context
    set(_map_prefix VID_DEBUG_RUN_TARGET)
  endif()

  if(${VIDSDK_HAS_VSP_TOOLCHAIN})
    # check if v-MP or v-SP mapping is defined for current run target
    if((NOT DEFINED ${_map_prefix}_${_target_upper}_MP_MAP) AND (NOT DEFINED ${_map_prefix}_${_target_upper}_SP_MAP))
      message(FATAL_ERROR "No v-MPs or v-SPs are mapped for current run target ${_target}. "
        "Call vid_map_build_target() to map videantis build targets to v-MPs or v-SPs")
    endif()

    # check if a v-MP map is defined
    if(DEFINED ${_map_prefix}_${_target_upper}_MP_MAP)
      # get the number of mapped v-MPs
      list(LENGTH ${_map_prefix}_${_target_upper}_MP_MAP _num_mps_mapped)
      # check if the number of mapped v-MPs matches the number of defined cores
      if(NOT ${_num_mps_mapped} EQUAL ${${_generic_prefix}_${_target_upper}_NUM_MPS})
        message(FATAL_ERROR "The number of mapped v-MPs matches not the number of defined cores in the simulator configuration")
      endif()
    endif()

    # check if a v-SP map is defined
    if(DEFINED ${_map_prefix}_${_target_upper}_SP_MAP)
      # get the number of mapped v-SPs
      list(LENGTH ${_map_prefix}_${_target_upper}_SP_MAP _num_sps_mapped)
      # check if the number of mapped v-SPs matches the number of defined cores
      if(NOT ${_num_sps_mapped} EQUAL ${${_generic_prefix}_${_target_upper}_NUM_SPS})
        message(FATAL_ERROR "The number of mapped v-SPs matches not the number of defined cores in the simulator configuration")
      endif()
    endif()

    # check if target cpus for v-SP and/or v-MP are defined
    if((NOT DEFINED ${_map_prefix}_${_target_upper}_TARGET_CPU_SP) AND (NOT DEFINED ${_map_prefix}_${_target_upper}_TARGET_CPU_MP))
      message(FATAL_ERROR "No v-SP and v-MP target cpu found for ${_target}")
    endif()
  else()
    # check if v-MP mapping is defined for current run target
    if(NOT DEFINED ${_map_prefix}_${_target_upper}_MP_MAP)
      message(FATAL_ERROR "No v-MPs are mapped for current run target ${_target}. "
        "Call vid_map_build_target() to map videantis build targets to v-MPs")
    endif()

    # get the number of mapped v-MPs
    list(LENGTH ${_map_prefix}_${_target_upper}_MP_MAP _num_mps_mapped)
    # check if the number of mapped v-MPs matches the number of defined cores
    if(NOT ${_num_mps_mapped} EQUAL ${${_generic_prefix}_${_target_upper}_NUM_MPS})
      message(FATAL_ERROR "The number of mapped v-MPs matches not the number of defined cores in the simulator configuration")
    endif()

    # check if target cpu for v-MP is defined
    if(NOT DEFINED ${_map_prefix}_${_target_upper}_TARGET_CPU_MP)
      message(FATAL_ERROR "No v-MP target cpu found for ${_target}")
    endif()
  endif()

  # check if target soc is defined
  if(NOT DEFINED ${_map_prefix}_${_target_upper}_TARGET_SOC)
    message(FATAL_ERROR "No target soc found for ${_target}")
  endif()

  # prepare variables to replace in simulator configuration template
  # variables will be redefined as generic onces (without run target name inside)
  # redefine map and bin file for each v-MP
  foreach(_mp ${${_map_prefix}_${_target_upper}_MP_MAP})
    set(${_generic_prefix}_BIN_FILE_MP${_mp} ${${_map_prefix}_${_target_upper}_BIN_FILE_MP${_mp}})
    set(${_generic_prefix}_MAP_FILE_MP${_mp} ${${_map_prefix}_${_target_upper}_MAP_FILE_MP${_mp}})
  endforeach(_mp)
  # redefine map and bin file for each v-SP
  foreach(_sp ${${_map_prefix}_${_target_upper}_SP_MAP})
    set(${_generic_prefix}_BIN_FILE_SP${_sp} ${${_map_prefix}_${_target_upper}_BIN_FILE_SP${_sp}})
    set(${_generic_prefix}_MAP_FILE_SP${_sp} ${${_map_prefix}_${_target_upper}_MAP_FILE_SP${_sp}})
  endforeach(_sp)
  # set current source dir
  set(${_generic_prefix}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  # set videantis run target dir
  set(${_generic_prefix}_DIR ${CMAKE_CURRENT_BINARY_DIR}/${_working_dir})
  if(IS_ABSOLUTE ${_working_dir})
    # set videantis run target dir
    set(${_generic_prefix}_DIR ${_working_dir})
  endif()
  # set run target name
  set(${_generic_prefix} ${_target})

  # if v-SP tool chain is available and a target cpu for v-SP is defined, set cpu v-SP
  if(${VIDSDK_HAS_VSP_TOOLCHAIN} AND (DEFINED ${_map_prefix}_${_target_upper}_TARGET_CPU_SP))
    set(${_generic_prefix}_CPU_SP ${${_map_prefix}_${_target_upper}_TARGET_CPU_SP})
  endif()
  # if a target cpu for v-MP is defined, set cpu v-MP
  if(DEFINED ${_map_prefix}_${_target_upper}_TARGET_CPU_MP)
    set(${_generic_prefix}_CPU_MP ${${_map_prefix}_${_target_upper}_TARGET_CPU_MP})
  endif()
  # set soc
  set(${_generic_prefix}_SOC ${${_map_prefix}_${_target_upper}_TARGET_SOC})

  # check if videantis simulator configuration template is defined
  if(NOT DEFINED ${_generic_prefix}_${_target_upper}_${_cfg_type}_TEMPLATE)
    message(FATAL_ERROR "Could not find a videantis simulator configuration template!")
  endif()
  # get directory from template (this includes the absolute path)
  get_filename_component(${_generic_prefix}_${_cfg_type}_PATH ${${_generic_prefix}_${_target_upper}_${_cfg_type}_TEMPLATE} DIRECTORY)
  # check if videantis simulator configuration template was defined with a absolute path
  if(NOT IS_ABSOLUTE ${${_generic_prefix}_${_cfg_type}_PATH})
    message(FATAL_ERROR "Could not determine absolute path of videantis simulator configuration template")
  endif()
endmacro(_prepare_cfg_variables)

#[=======================================================================[.rst:
.. cmake:command:: vid_run_target

  Macro for running a one or more videantis build targets.
  A build target can be run standalone (videantis simulator only) or with a host application.
  The host application needs to be build via cmake and can be forked by the videantis
  simulator (if supported) or the host needs to fork the videantis simulator. Input dependencies
  to the run target can be specified via INPUTS and generated output files via OUTPUTS.
  The generated output from the videantis simulator will be detected automatically from the
  simulator configuration and does not need to specified by hand.
  Every run target will be run in sub directory of the current cmake build directory.
  This ensures that multiple run configuration do not conflict each other and run results/outputs
  are sorted by default. The run directory is always run_<target_name> (if parameter WORKING_DIRECTORY
  is not defined) and will be reported in the variable VID_RUN_TARGET_<TARGET_NAME>_WORKING_DIR
  after the macro :cmake:command:`vid_run_target` is called.

  .. code-block:: cmake

    vid_run_target(target [EXTENSION extension] [FORK_VIDSIM target] [FORK_HOST target]
        [HOST_ARGS arg1 [arg2 ...]] [INPUTS file1 [file2 ...]] [OUTPUTS file1 [file2 ...]]
        [WORKING_DIRECTORY dir] [NO_WARN_HOST_ARGS] [NO_WARM_MISSING_EXT_FILES]
        [DEBUG_HOST] [NO_TARGET])

  Function parameters:

  ``target``
    name of an already defined run target
  ``EXTENSION``
    define videantis simulator extension (cct or man)
  ``FORK_VIDSIM``
    fork videantis simulator with host application (cmake host application target needs to be set)
  ``FORK_HOST``
    fork host application with videantis simulator (cmake host application target needs to be set)
  ``HOST_ARGS``
    arguments for a host application
  ``INPUTS``
    required input files
  ``OUTPUTS``
    generated output files
  ``WORKING_DIRECTORY``
    define working directory (default: run_<target_name> in current cmake binary directory)
  ``NO_WARN_HOST_ARGS``
    silence warning when HOST_ARGS are defined and arguments for the host
    application provided with -host_app_args in the simulator configuration
  ``NO_WARM_MISSING_EXT_FILES``
    silence warning when output file of videantis simulator
    extension is not defined in simulator configuration
  ``DEBUG_HOST``
    start videantis simulator only (even when FORK_HOST or FORK_VIDSIM is present) and
    print the path and command line arguments to start the host application for debugging
  ``NO_TARGET``
    create no cmake targets

  Generic variables available for replacement in videantis simulator configuration:

  ``VID_RUN_TARGET``
    name of the run target
  ``VID_RUN_TARGET_SOURCE_DIR``
    full path to source dir of run target
  ``VID_RUN_TARGET_DIR``
    full path to dir of run target
  ``VID_RUN_TARGET_CPU_MP``
    target cpu v-MP
  ``VID_RUN_TARGET_CPU_SP``
    target cpu v-SP
  ``VID_RUN_TARGET_SOC``
    target soc
  ``VID_RUN_TARGET_BIN_FILE_MP<MP_ID>``
    full path to bin file for v-MP core
  ``VID_RUN_TARGET_MAP_FILE_MP<MP_ID>``
    full path to map file for v-MP core
  ``VID_RUN_TARGET_BIN_FILE_SP<SP_ID>``
    full path to bin file for v-SP core
  ``VID_RUN_TARGET_MAP_FILE_SP<SP_ID>``
    full path to map file for v-SP core
  ``VID_RUN_TARGET_SIMCFG_PATH``
    full path to folder of simulator configuration template

  Exposed variables:

  ``VID_RUN_TARGET_<TARGET_NAME>_WORKING_DIR``
    working directory for run target (all artifacts will be written into this directory)
  ``VID_RUN_TARGET_<TARGET_NAME>_INPUTS``
    list of inputs of the run target
  ``VID_RUN_TARGET_<TARGET_NAME>_GENERATED_INPUTS``
    list of generated inputs by build targets of the run target
  ``VID_RUN_TARGET_<TARGET_NAME>_OUTPUTS``
    list of outputs of the run target
  ``VID_RUN_TARGET_<TARGET_NAME>_COMMANDS``
    list of commands of the run target

  Exposed cmake targets:

  ``run_<target_name>``
    run target

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<target_name>``
    target name
  ``<MP_ID>``
    ID of a v-MP core
  ``<SP_ID>``
    ID of a v-SP core
#]=======================================================================]
macro(vid_run_target _target)
  cmake_parse_arguments(_ARG "NO_WARN_HOST_ARGS;NO_WARN_MISSING_EXT_FILES;DEBUG_HOST;NO_TARGET"
    "FORK_VIDSIM;FORK_HOST;EXTENSION;WORKING_DIRECTORY" "HOST_ARGS;INPUTS;OUTPUTS" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if videantis run target is defined
  if(NOT DEFINED VID_RUN_TARGET_${_target_upper})
    message(FATAL_ERROR "Run target ${_target} is not found in videantis build context")
  endif()

  # check if this run target is debug only
  if(${VID_RUN_TARGET_${_target_upper}_DEBUG_ONLY})
    message(FATAL_ERROR "This videantis run target is initialized as debug only. Calling vid_run_target() is prohibited")
  endif()

  set(_tool VIDSIM)
  if(DEFINED _ARG_EXTENSION)
    # create all upper case string of EXTENSION
    string(TOUPPER ${_ARG_EXTENSION} _extension_upper)
    # check if extension defines a valid vidsim extension (cct or man)
    if((NOT ${_extension_upper} STREQUAL "CCT") AND (NOT ${_extension_upper} STREQUAL "MAN"))
      message(FATAL_ERROR "Only videantis simulator code coverage (cct) and memory analyzer extensions (man) are supported")
    endif()

    # create a updated tool variable for vidsim extension
    set(_tool VIDSIM-${_extension_upper})
  endif()

  # TODO: check why unparsed arguments is never set
  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_run_target() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # check if only forking vidsim from host or forking host from vidsim is specified
  if((DEFINED _ARG_FORK_VIDSIM) AND (DEFINED _ARG_FORK_HOST))
    message(FATAL_ERROR "Arguments FORK_HOST and FORK_VIDSIM cannot defined at the same call")
  endif()

  # check if forking host application from vidsim is specified and vidsim supports this feature
  if((DEFINED _ARG_FORK_HOST) AND (NOT ${${_tool}_HAS_HOST_APP}))
    message(FATAL_ERROR "The version of the videantis simulator included in this SDK "
      "does not support forking the host application")
  endif()

  # check if argument DEBUG_HOST makes sense in the actual configuration
  if((NOT DEFINED _ARG_FORK_HOST) AND (NOT DEFINED _ARG_FORK_VIDSIM) AND ${_ARG_DEBUG_HOST})
    message(WARNING "Argument DEBUG_HOST is defined when calling vid_run_target(), "
      "but no host application is set via FORK_HOST or FORK_VIDSIM. Argument will be ignored.")
  endif()

  # set working dir for run target
  set(VID_RUN_TARGET_${_target_upper}_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/run_${_target})
  # set default working directory (relative path of CMAKE_CURRENT_BINARY_DIR)
  set(_working_dir run_${_target})
  # if a parameter WORKING_DIRECTORY is specified
  if(DEFINED _ARG_WORKING_DIRECTORY)
    # normalize provided path (see: https://cmake.org/cmake/help/latest/command/cmake_path.html#normalization)
    cmake_path(NORMAL_PATH _ARG_WORKING_DIRECTORY OUTPUT_VARIABLE _working_dir)
    # remove slash if last character of the path is a slash and more characters then a slash are in the string
    string(REGEX REPLACE "(.+)/$" "\\1" _working_dir ${_working_dir})

    # if the specified working directory is an absolute path
    if(IS_ABSOLUTE ${_working_dir})
      # check if specified working directory is part of the build directory
      cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${_working_dir} _wd_in_bd)
      if(NOT ${_wd_in_bd})
        message(FATAL_ERROR "WORKING_DIRECTORY has to be a directory inside the build directory!")
      endif()
      # overwrite default working dir for run target
      set(VID_RUN_TARGET_${_target_upper}_WORKING_DIR ${_working_dir})
    else()
      # overwrite default working dir for run target
      set(VID_RUN_TARGET_${_target_upper}_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/${_working_dir})
    endif()
  endif()

  # clear internal variables from previous calls of vid_run_target()
  unset(_host_target)
  unset(_run_target_outputs)
  unset(_vidsim_args)
  unset(_vidsim_output_redirect)
  unset(_host_output_redirect)
  unset(_debug_host_print_target)
  unset(_map_bin_files)

  # set default value for target_has_host to FALSE
  set(_target_has_host FALSE)
  # check if FORK_VIDSIM or FORK_HOST is defined
  if((DEFINED _ARG_FORK_VIDSIM) OR (DEFINED _ARG_FORK_HOST))
    # set name of host target
    # only one argument is defined FORK_HOST or FORK_VIDSIM (checked before)
    set(_host_target ${_ARG_FORK_HOST} ${_ARG_FORK_VIDSIM})
    # check if host_target is a valid cmake target
    if(TARGET ${_host_target})
      set(_target_has_host TRUE)
    else()
      message(FATAL_ERROR "Defined host target ${_host_target} is not available")
    endif()
  endif()

  # check if host application for simulation is defined and HOST_ARGS are defined
  # raise warning to make aware that HOST_ARGS will be ignored
  if((NOT ${_target_has_host}) AND (DEFINED _ARG_HOST_ARGS))
    message(WARNING "HOST_ARGS is specified when calling macro vid_run_target(), "
      "but no host application is defined via FORK_HOST or FORK_VIDSIM. "
      "HOST_ARGS will be ignored")
  endif()

  # prepare variables to be inserted into simulator configuration template
  _prepare_cfg_variables(${_target} run ${_working_dir})

  # prepare simulator configuration for simulation
  set(_simcfg vid_sim_config_link.cfg)

  # output: _FILE_CONTENT
  vid_cached_configure_file(${VID_RUN_TARGET_${_target_upper}_SIMCFG_TEMPLATE} ${VID_RUN_TARGET_${_target_upper}_WORKING_DIR}/${_simcfg})

  # add generated simcfg to run targets inputs
  set(_run_target_inputs ${VID_RUN_TARGET_${_target_upper}_WORKING_DIR}/${_simcfg})

  # parse the resolved content directly in memory (avoids writing to disk and re-reading the
  # unique generated file per test)
  # output: _SIMCFG_LIST
  vid_parse_simcfg_as_list(FROM_STRING "${_FILE_CONTENT}")

  # parse the simcfg for "-f" parameter
  # output: _SIMCFG_ARG
  vid_get_arg_from_simcfg_list(-f SIMCFG ${_SIMCFG_LIST} MULTIPLE)
  # iterate over found parameter files
  foreach(_arg ${_SIMCFG_ARG})
    # check if absolute path is defined/generated and the file exists
    if((IS_ABSOLUTE ${_arg}) AND (EXISTS ${_arg}))
      # append parameter file to run target inputs
      # this ensures cmake runs again without user interaction, if the file is changed
      list(APPEND _run_target_inputs ${_arg})
    else()
      message(FATAL_ERROR "The parameter file found in the generated simcfg does not exists or has no absolute path: ${_arg} "
        "NOTE: Best practice is to define the path to the file as absolute path with the help of @VID_RUN_TARGET_SIMCFG_PATH@")
    endif()
  endforeach(_arg)

  # get simulator inputs from generated simcfg
  # iterate over simulator arguments that require input files
  foreach(_input_arg ${${_tool}_INPUT_ARGS})
    # set default position for filenames in input argument
    set(_pos 0)
    # -cct_user_html defines the filename at position 1
    if(${_input_arg} STREQUAL "-cct_user_html")
      set(_pos 1)
    endif()
    # parse current arguments from simcfg
    # (multiple definitions will be parsed; filenames are always on position 0 for every argument except -cct_user_html (pos 1))
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(${_input_arg} POS ${_pos} SIMCFG ${_SIMCFG_LIST} MULTIPLE)
    # iterate over found files
    foreach(_file ${_SIMCFG_ARG})
      # append parsed argument to run target inputs list
      list(APPEND _run_target_inputs ${_file})
    endforeach(_file)
  endforeach(_input_arg)

  # get simulator outputs from generated simcfg
  # iterate over simulator arguments that generate output files
  foreach(_output_arg ${${_tool}_OUTPUT_ARGS})
    # parse current arguments from simcfg
    # (multiple definitions will be parsed; filenames are always on position 0 for every argument)
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(${_output_arg} POS 0 SIMCFG ${_SIMCFG_LIST} MULTIPLE)
    # iterate over found files
    foreach(_file ${_SIMCFG_ARG})
      # check if provided output has an absolute path
      if(IS_ABSOLUTE ${_file})
        # append output with absolute path from <TOOL>_OUTPUT_ARGS to run target outputs
        list(APPEND _run_target_outputs ${_file})
      else()
        # if working directory is current binary directory
        if(${_working_dir} STREQUAL ".")
          # append output from TOOL>_OUTPUT_ARGS  to run target outputs
          list(APPEND _run_target_outputs ${_file})
        else()
          # append output from TOOL>_OUTPUT_ARGS to run target outputs with working dir path
          list(APPEND _run_target_outputs ${_working_dir}/${_file})
        endif()
      endif()
    endforeach(_file)
  endforeach(_output_arg)

  # append INPUTS to run target inputs
  # (only if defined otherwise nothing will be appended)
  list(APPEND _run_target_inputs ${_ARG_INPUTS})
  # iterate over _ARG_OUTPUTS
  # (only if defined otherwise nothing will be appended)
  foreach(_output ${_ARG_OUTPUTS})
    # check if provided output has an absolute path
    if(IS_ABSOLUTE ${_output})
      # append output with absolute path from _ARG_OUTPUTS to run target outputs
      list(APPEND _run_target_outputs ${_output})
    else()
      # if working directory is current binary directory
      if(${_working_dir} STREQUAL ".")
        # append output from _ARG_OUTPUTS to run target outputs
        list(APPEND _run_target_outputs ${_output})
      else()
        # append output from _ARG_OUTPUTS to run target outputs with working dir path
        list(APPEND _run_target_outputs ${_working_dir}/${_output})
      endif()
    endif()
  endforeach(_output)
  # remove possible duplicates from run target inputs and outputs lists
  list(REMOVE_DUPLICATES _run_target_inputs)
  list(REMOVE_DUPLICATES _run_target_outputs)

  # add all v-MP map and bin files to a list
  foreach(_mp ${VID_RUN_TARGET_${_target_upper}_MP_MAP})
    list(APPEND _map_bin_files ${VID_RUN_TARGET_${_target_upper}_MAP_FILE_MP${_mp}} ${VID_RUN_TARGET_${_target_upper}_BIN_FILE_MP${_mp}})
  endforeach(_mp)
  # add all v-SP map and bin files to a list (if v-SP toolchain is included)
  foreach(_sp ${VID_RUN_TARGET_${_target_upper}_SP_MAP})
    list(APPEND _map_bin_files ${VID_RUN_TARGET_${_target_upper}_MAP_FILE_SP${_sp}} ${VID_RUN_TARGET_${_target_upper}_BIN_FILE_SP${_sp}})
  endforeach(_sp)

  # check if simcfg has parameter "-host_app" set
  # output: _SIMCFG_ARG
  vid_get_arg_from_simcfg_list(-host_app SIMCFG ${_SIMCFG_LIST} IS_SET)
  # if "-host_app" is set raise error
  if(${_SIMCFG_ARG})
    message(FATAL_ERROR "Simulator config has the parameter \"-host_app\" set. "
      "This parameter needs to be set from cmake for forking the host application with vidsim. "
      "For all other operation modes it needs to be unset as well")
  endif()

  # preparation for running a host application
  if(${_target_has_host})
    # do additional preparation in the case the host application should be forked by the simulator
    # and the host application should not be debugged
    if((DEFINED _ARG_FORK_HOST) AND (NOT ${_ARG_DEBUG_HOST}))
      # transfer the list with host arguments from macro call into a whitespace separated string
      string(REPLACE ";" " " _host_args_spaced "${_ARG_HOST_ARGS}")
      # create stripped argument for -host_app parameter
      string(STRIP "$<TARGET_FILE:${_host_target}> ${_host_args_spaced}" _host_app)
      # define vidsim arguments list
      set(_vidsim_args -host_app ${_host_app})

      # check if simcfg has parameter "-host_app_stdout" set
      # output: _SIMCFG_ARG
      vid_get_arg_from_simcfg_list(-host_app_stdout SIMCFG ${_SIMCFG_LIST} IS_SET)
      # if "-host_app_stdout" is not set
      # and host application stdout should not be visible on stdout
      # in the case "-host_app_stdout" is set in the simcfg, the output will be added by default to cmake outputs
      if((NOT ${_SIMCFG_ARG}) AND (NOT ${VIDSDK_HOST_STDOUT}))
        # append "-host_app_stdout" to vidsim arguments
        list(APPEND _vidsim_args -host_app_stdout ${_target}.host.txt)
        # if working directory is current binary directory
        if(${_working_dir} STREQUAL ".")
          # add forked host application stdout file to run target outputs
          list(APPEND _run_target_outputs ${_target}.host.txt)
        else()
          # add forked host application stdout file to run target outputs
          list(APPEND _run_target_outputs ${_working_dir}/${_target}.host.txt)
        endif()
      # raise warning if "-host_stdout" parameter is set in simcfg
      # and host application stdout should be visible on stdout
      elseif(${_SIMCFG_ARG} AND ${VIDSDK_HOST_STDOUT})
        message(WARNING "CMake option VIDSDK_HOST_STDOUT is enabled, but the simulator configuration has \"-host_stdout\" set. "
          "Change simulator configuration template to see host application output on stdout: ${VID_RUN_TARGET_${_target_upper}_SIMCFG_TEMPLATE}")
      endif()

      # check if simcfg has parameter "-host_app_args" set and HORST_ARGS is specified in macro call
      # output: _SIMCFG_ARG
      vid_get_arg_from_simcfg_list(-host_app_args SIMCFG ${_SIMCFG_LIST} IS_SET)
      # warning can be silenced when calling macro with NO_WARN_HOST_ARGS
      if(${_SIMCFG_ARG} AND (DEFINED _ARG_HOST_ARGS) AND (NOT ${_ARG_NO_WARN_HOST_ARGS}))
        message(WARNING "Macro vid_run_target() is called with HOST_ARGS and specified the simulator "
          "configuration has \"-host_app_args\" set. \"-host_app_args\" will be appended to the HOST_ARGS. "
          "This warning can be silenced by calling macro vid_run_target() with option NO_WARN_HOST_ARGS")
      endif()
    endif()
  endif()

  # check if vidsim extension is cct and cct specific files are defined in simcfg
  if(${_tool} STREQUAL "VIDSIM-CCT")
    # check if simcfg has parameter "-cct_report" set
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(-cct_report SIMCFG ${_SIMCFG_LIST} IS_SET)
    set(_cct_report_is_set ${_SIMCFG_ARG})
    # check if simcfg has parameter "-cct_data" set
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(-cct_data SIMCFG ${_SIMCFG_LIST} IS_SET)
    set(_cct_data_is_set ${_SIMCFG_ARG})
    if((NOT (_cct_report_is_set AND _cct_data_is_set)) AND (NOT _ARG_NO_WARN_MISSING_EXT_FILES))
      message(WARNING "videantis simulator code coverage extension is used without defining -cct_report and -cct_data files in simulator configuration")
    endif()

    # if cct data file is available in configuration and the run target should create targets
    if(${_cct_data_is_set} AND (NOT ${_ARG_NO_TARGET}))
      # get parameter "-cct_data" from simcfg
      # output: _SIMCFG_ARG
      vid_get_arg_from_simcfg_list(-cct_data POS 0 SIMCFG ${_SIMCFG_LIST})
      set(_cct_data_file ${_SIMCFG_ARG})

      # define dummy file to have a timestamped file when the last cct data file deletion has happened
      set(_cct_data_file_watcher_wo_wdir last_cct_data_deletion_${_target})
      set(_cct_data_file_watcher ${_cct_data_file_watcher_wo_wdir})
      # if working directory is not the current binary directory
      if(NOT ${_working_dir} STREQUAL ".")
        # prepend working directory to dummy file
        string(PREPEND _cct_data_file_watcher ${_working_dir}/)
      endif()

      # define custom command that remove cct data file, if bin and map files have changed
      # the command creates an empty dummy to monitor the last deletion and only if the
      # bin and map file are newer then the dummy file, the cct data file will be deleted again
      add_custom_command(OUTPUT ${_cct_data_file_watcher}
        COMMAND ${CMAKE_COMMAND} -E rm -f ${_cct_data_file}
        COMMAND ${CMAKE_COMMAND} -E touch ${_cct_data_file_watcher_wo_wdir}
        DEPENDS ${_map_bin_files}
        WORKING_DIRECTORY ${_working_dir}
        COMMENT "Removing old cct data file ${_cct_data_file} ...")

      # append cct data file watcher file to run target inputs list
      list(APPEND _run_target_inputs ${_cct_data_file_watcher})
    endif()
  endif()

  # check if vidsim extension is man and man specific file ist defined in simcfg
  if(${_tool} STREQUAL "VIDSIM-MAN")
    # check if simcfg has parameter "-man" set
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(-man SIMCFG ${_SIMCFG_LIST} IS_SET)
    if((NOT ${_SIMCFG_ARG}) AND (NOT _ARG_NO_WARN_MISSING_EXT_FILES))
      message(WARNING "videantis simulator memory analyzer extension is used without defining -man file in simulator configuration")
    endif()
  endif()

  # check if vidsim should be forked by host application
  # and the host application should not be debugged
  # otherwise run vidsim and optionally fork host application
  if((DEFINED _ARG_FORK_VIDSIM) AND (NOT ${_ARG_DEBUG_HOST}))
    # set log file for host stdout
    set(_host_out_file ${_target}.host.txt)
    # check if host application stdout should be visible on stdout
    if(${VIDSDK_HOST_STDOUT})
      # define redirecting the output of host application to log file and stdout
      set(_host_output_redirect | tee ${_host_out_file})
    else()
      # define redirecting the output of host application to log file
      set(_host_output_redirect > ${_host_out_file})
    endif()
    # if working directory is current binary directory
    if(${_working_dir} STREQUAL ".")
      # add log file of host stdout file list of run target outputs
      list(APPEND _run_target_outputs ${_host_out_file})
    else()
      # add log file of host stdout file list of run target outputs
      list(APPEND _run_target_outputs ${_working_dir}/${_host_out_file})
    endif()

    # set run target commands list
    set(VID_RUN_TARGET_${_target_upper}_COMMANDS $<TARGET_FILE:${_host_target}> ${_ARG_HOST_ARGS} ${_host_output_redirect})

    # check if a target should be created
    if(NOT ${_ARG_NO_TARGET})
      # define custom command to run the host application
      add_custom_command(OUTPUT ${_run_target_outputs}
        COMMAND $<TARGET_FILE:${_host_target}> ${_ARG_HOST_ARGS} ${_host_output_redirect}
        DEPENDS ${_run_target_inputs} ${_host_target}
        WORKING_DIRECTORY ${_working_dir}
        VERBATIM COMMENT "Running ${_target} in ${VID_RUN_TARGET_${_target_upper}_WORKING_DIR} ...")
    endif()
  else()
    # if vidsim supports "-log" parameter
    if(${${_tool}_HAS_LOG})
      # check if simcfg has parameter "-log" set
      # output: _SIMCFG_ARG
      vid_get_arg_from_simcfg_list(-log SIMCFG $_SIMCFG_LIST} IS_SET)
      # check if simcfg parameter "-log" it not set
      # and simulator stdout should not be visible on stdout
      if((NOT ${_SIMCFG_ARG}) AND (NOT ${VIDSDK_SIM_STDOUT}))
        # set log file for vidsim stdout
        set(_vidsim_out_file ${_target}.vidsim.txt)
        # append "-log" parameter plus vidsim stdout file to vidsim arguments
        list(APPEND _vidsim_args -log ${_vidsim_out_file})
        # if working directory is current binary directory
        if(${_working_dir} STREQUAL ".")
          # add log file of vidsim stdout to file list of run target outputs
          list(APPEND _run_target_outputs ${_vidsim_out_file})
        else()
          # add log file of vidsim stdout to file list of run target outputs
          list(APPEND _run_target_outputs ${_working_dir}/${_vidsim_out_file})
        endif()
      # check if simcfg parameter "-log" it not set
      # and simulator stdout should be visible on stdout
      elseif((NOT ${_SIMCFG_ARG}) AND ${VIDSDK_SIM_STDOUT})
        # set log file for vidsim stdout
        set(_vidsim_out_file ${_target}.vidsim.txt)
        # define redirecting the output of vidsim to log file
        set(_vidsim_output_redirect | tee ${_vidsim_out_file})
        # if working directory is current binary directory
        if(${_working_dir} STREQUAL ".")
          # add log file of vidsim stdout to file list of run target outputs
          list(APPEND _run_target_outputs ${_vidsim_out_file})
        else()
          # add log file of vidsim stdout to file list of run target outputs
          list(APPEND _run_target_outputs ${_working_dir}/${_vidsim_out_file})
        endif()
      # raise warning if "-log" parameter is set in simcfg
      # and simulator stdout should be visible on stdout
      elseif(${_SIMCFG_ARG} AND ${VIDSDK_SIM_STDOUT})
        message(WARNING "CMake option VIDSDK_SIM_STDOUT is enabled, but the simulator configuration has \"-log\" set. "
          "Change simulator configuration template to see simulator output on stdout: ${VID_RUN_TARGET_${_target_upper}_SIMCFG_TEMPLATE}")
      endif()
    # vidsim does not support "-log" parameter
    else()
      # set log file for vidsim stdout
      set(_vidsim_out_file ${_target}.vidsim.txt)
      # check if simulator stdout should be visible on stdout
      if(${VIDSDK_SIM_STDOUT})
        # define redirecting the output of vidsim to log file and stdout
        set(_vidsim_output_redirect | tee ${_vidsim_out_file})
      else()
        # define redirecting the output of vidsim to log file
        set(_vidsim_output_redirect > ${_vidsim_out_file})
      endif()
      # if working directory is current binary directory
      if(${_working_dir} STREQUAL ".")
        # add log file of vidsim stdout to file list of run target outputs
        list(APPEND _run_target_outputs ${_vidsim_out_file})
      else()
        # add log file of vidsim stdout to file list of run target outputs
        list(APPEND _run_target_outputs ${_working_dir}/${_vidsim_out_file})
      endif()
    endif()

    if(${_ARG_DEBUG_HOST} AND ${_target_has_host} AND (NOT ${_ARG_NO_TARGET}))
      # set name for print debug host information target
      set(_debug_host_print_target run_${_target}_print_debug_host)
      # transfer the list with host arguments from macro call into a whitespace separated string
      string(REPLACE ";" " " _host_args_spaced "${_ARG_HOST_ARGS}")
      # check if forking vidsim by host application is defined
      if(DEFINED _ARG_FORK_VIDSIM)
        # define custom target to print information for debugging host application for the case vidsim should be forked by the host application
        add_custom_target(${_debug_host_print_target}
          COMMAND ${CMAKE_COMMAND} -E echo "Start host application in debug environment as following: $<TARGET_FILE:${_host_target}> ${_host_args_spaced}"
          COMMAND ${CMAKE_COMMAND} -E echo "Note: Please ensure that none of the arguments above enables forking videantis simulator by the host application. Cmake will start the videantis simulator ..."
          DEPENDS ${_run_target_inputs} ${_host_target}
          VERBATIM COMMENT "Printing information to debug host application ...")
      else()
        # define custom target to print information for debugging host application
        add_custom_target(${_debug_host_print_target}
          COMMAND ${CMAKE_COMMAND} -E echo "Start host application in debug environment as following: $<TARGET_FILE:${_host_target}> ${_host_args_spaced}"
          DEPENDS ${_run_target_inputs} ${_host_target}
          VERBATIM COMMENT "Printing information to debug host application ...")
      endif()
    endif()

    # set run target commands list
    set(VID_RUN_TARGET_${_target_upper}_COMMANDS ${${_tool}_EXECUTABLE} ${_vidsim_args} -f ${_simcfg} ${_vidsim_output_redirect})

    # check if a target should be created
    if(NOT ${_ARG_NO_TARGET})
      # define custom command to run the simulator
      add_custom_command(OUTPUT ${_run_target_outputs}
        COMMAND ${${_tool}_EXECUTABLE} ${_vidsim_args} -f ${_simcfg} ${_vidsim_output_redirect}
        DEPENDS ${_run_target_inputs} ${_host_target} ${_debug_host_print_target}
        WORKING_DIRECTORY ${_working_dir}
        VERBATIM COMMENT "Running ${_target} in ${VID_RUN_TARGET_${_target_upper}_WORKING_DIR} ...")
    endif()
  endif()

  # set run target inputs list
  set(VID_RUN_TARGET_${_target_upper}_INPUTS ${_run_target_inputs})
  # remove generated inputs from mapped build targets from run targets inputs list
  list(REMOVE_ITEM VID_RUN_TARGET_${_target_upper}_INPUTS ${_map_bin_files})
  # set run target inputs list with generated inputs from mapped build targets
  set(VID_RUN_TARGET_${_target_upper}_GENERATED_INPUTS ${_map_bin_files})
  # set run target outputs list
  set(VID_RUN_TARGET_${_target_upper}_OUTPUTS ${_run_target_outputs})

  # check if a target should be created
  if(NOT ${_ARG_NO_TARGET})
    # define custom target that depends on run target outputs
    add_custom_target(run_${_target}
      DEPENDS ${_run_target_outputs})
  endif()
endmacro(vid_run_target)

#[=======================================================================[.rst:
.. cmake:command:: vid_debug_run_target

  Macro for debugging a one or more videantis build targets.
  A build target can be debugged standalone (videantis debugger only) or with a host application.
  The host application needs to be build via cmake and can be forked by the videantis
  debugger (if supported) or the host will be started and videantis debugger needs to be started
  separately. Input dependencies to the debug target can be specified via INPUTS and generated
  output files via OUTPUTS. The generated output from the videantis debugger will be detected
  automatically from the debugger configuration and does not need to specified by hand.
  Every debug target will be run in sub directory of the current cmake build directory.
  This ensures that multiple debug or run configurations do not conflict each other and
  debugging results/outputs are sorted by default. The debug directory is always vdebug_<target_name>
  (if parameter WORKING_DIRECTORY is not defined) and will be reported in the variable
  VID_DEBUG_RUN_TARGET_<TARGET_NAME>_WORKING_DIR after the macro :cmake:command:`vid_debug_run_target`
  is called. To ensure the outputs of the videantis debugger are written into the correct directory it
  is best practice to set in the debugger configuration template the "-cd" to @VID_DEBUG_RUN_TARGET_DIR@.
  Cmake will generate a correct path for this variable. Every debug directory contains a
  <target_name>.dbgcfg file with a debugger configuration based on the defined debugger configuration
  template. If a host application is used, an additional vid_sim_config_link.cfg configuration file is
  created as a symlink based on <target_name>.dbgcfg. Running a debug cmake target always creates a copy
  of the <target_name>.dbgcfg depending of the cmake target in the root of the cmake binary directory
  named <project_name>.dbgcfg.

  .. code-block:: cmake

    vid_debug_run_target(target [DBGCFG_TEMPLATE template] [FORK_HOST]
        [HOST_APP target] [HOST_ARGS arg1 [arg2 ...]] [INPUTS file1 [file2 ...]]
        [OUTPUTS file1 [file2 ...]] [WORKING_DIRECTORY dir] [NO_WARN_HOST_ARGS])

  Function parameters:

  ``target``
    name of an already defined run target
  ``DBGCFG_TEMPLATE``
    path to a debugger configuration template

  .. note::

    - If DBGCFG_TEMPLATE is not defined, the macro will search for a dbgcfg template named <target_name>.dbgcfg.in
    - If macro :cmake:command:`vid_init_run_target` has already DBGCFG_TEMPLATE set, the template defined with :cmake:command:`vid_debug_run_target` will be ignored

  ``FORK_HOST``
    fork host application with videantis debugger
  ``HOST_APP``
    cmake host application target
  ``HOST_ARGS``
    arguments for a host application
  ``INPUTS``
    required input files
  ``OUTPUTS``
    generated output files
  ``WORKING_DIRECTORY``
    define working directory (default: vdebug_<target_name> in current cmake binary directory)
  ``NO_WARN_HOST_ARGS``
    silence warning when HOST_ARGS are defined and arguments for the host
    application provided with -host_app_args in the simulator configuration

  Generic variables available for replacement in videantis debugger configuration:

  ``VID_DEBUG_RUN_TARGET``
    name of the run target to debug
  ``VID_DEBUG_RUN_TARGET_SOURCE_DIR``
    full path to source dir of run target to debug
  ``VID_DEBUG_RUN_TARGET_DIR``
    full path to dir of run target to debug
  ``VID_DEBUG_RUN_TARGET_CPU_MP``
    target cpu v-MP
  ``VID_DEBUG_RUN_TARGET_CPU_SP``
    target cpu v-SP
  ``VID_DEBUG_RUN_TARGET_SOC``
    target soc
  ``VID_DEBUG_RUN_TARGET_BIN_FILE_MP<MP_ID>``
    full path to bin file for v-MP core
  ``VID_DEBUG_RUN_TARGET_MAP_FILE_MP<MP_ID>``
    full path to map file for v-MP core
  ``VID_DEBUG_RUN_TARGET_BIN_FILE_SP<SP_ID>``
    full path to bin file for v-SP core
  ``VID_DEBUG_RUN_TARGET_MAP_FILE_SP<SP_ID>``
    full path to map file for v-SP core
  ``VID_DEBUG_RUN_TARGET_DBGCFG_PATH``
    full path to folder of simulator configuration template
  ``VID_DEBUG_RUN_TARGET_HOST_APP``
    full path to host binaries including host arguments
  ``VID_DEBUG_RUN_TARGET_HOST_STDOUT``
    filename for host stdout

  Exposed variables:

  ``VID_DEBUG_RUN_TARGET_<TARGET_NAME>_WORKING_DIR``
    working directory for run target to debug (all artifacts will be written to this dir)

  Exposed cmake targets:

  ``debug_<target_name>``
    debug target

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<target_name>``
    target name
  ``<project_name>``
    name of the cmake project in all lower case letters
  ``<MP_ID>``
    ID of a v-MP core
  ``<SP_ID>``
    ID of a v-SP core
#]=======================================================================]
macro(vid_debug_run_target _target)
  cmake_parse_arguments(_ARG "FORK_HOST;NO_WARN_HOST_ARGS" "DBGCFG_TEMPLATE;HOST_APP;WORKING_DIRECTORY"
    "HOST_ARGS;INPUTS;OUTPUTS" ${ARGN})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_debug_run_target() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if videantis run target is defined
  if(NOT DEFINED VID_RUN_TARGET_${_target_upper})
    message(FATAL_ERROR "Run target ${_target} is not found in videantis build context")
  endif()

  # check if forking host application from vidsim-dbg is specified and vidsim-dbg supports this feature
  if(${_ARG_FORK_HOST} AND (NOT ${VIDSIM-DBG_HAS_HOST_APP}))
    message(FATAL_ERROR "The version of the videantis debugger included in this SDK "
      "does not support forking the host application")
  endif()

  # check if forking host application from vidsim-dbg is specified and a host application is provided
  if(${_ARG_FORK_HOST} AND (NOT DEFINED _ARG_HOST_APP))
    message(FATAL_ERROR "The videantis debugger cannot fork the host application, "
      "because no host application is set via argument HOST_APP")
  endif()

  # set default map prefix for getting generated
  set(_map_prefix VID_RUN_TARGET)
  # check if debug only run target
  if(${VID_RUN_TARGET_${_target_upper}_DEBUG_ONLY})
    # overwrite map prefix for debug context
    set(_map_prefix VID_DEBUG_RUN_TARGET)
  endif()

  # check if a dbgcfg was already defined when calling vid_init_run_target()
  if(NOT DEFINED VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE)
    # if DBGCFG_TEMPLATE is defined as argument
    if(DEFINED _ARG_DBGCFG_TEMPLATE)
      # set DBGCFG_TEMPLATE as debugger configuration template file
      set(VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE ${_ARG_DBGCFG_TEMPLATE})
    else()
      # set default debugger configuration template file
      set(VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE ${_target}.dbgcfg.in)
      message(WARNING "If DBGCFG_TEMPLATE is not defined with vid_debug_run_target(), "
        "a default dbgcfg template will be assumed: ${VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE}")
    endif()

    # if dbgcfg_template is not a absolute path
    if(NOT IS_ABSOLUTE ${VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE})
      # create absolute path from relative path of dbgcfg_template
      get_filename_component(VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE ${VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE} ABSOLUTE)
    endif()

    # check if debugger configuration template file exists
    if(NOT EXISTS ${VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE})
      message(FATAL_ERROR "Debugger configuration template file ${VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE} could not found")
    endif()

    # get number of cores from dbgcfg template for debug context
    _get_num_of_cores(${_target} ${VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE} debug)
  else()
    # dbgcfg is already initialized and also set when calling vid_debug_run_target()
    if(DEFINED _ARG_DBGCFG_TEMPLATE)
      message(WARNING "When initializing this run target a debugger configuration template was already defined. "
        "The debugger configuration provided via argument DBGCFG_TEMPLATE when calling vid_debug_run_target() will be ignored")
    endif()
  endif()

  # set working dir for debug run target
  set(VID_DEBUG_RUN_TARGET_${_target_upper}_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/vdebug_${_target})
  # set default working directory (relative path of CMAKE_CURRENT_BINARY_DIR)
  set(_working_dir vdebug_${_target})
  # if a parameter WORKING_DIRECTORY is specified
  if(DEFINED _ARG_WORKING_DIRECTORY)
    # normalize provided path (see: https://cmake.org/cmake/help/latest/command/cmake_path.html#normalization)
    cmake_path(NORMAL_PATH _ARG_WORKING_DIRECTORY OUTPUT_VARIABLE _working_dir)
    # remove slash if last character of the path is a slash and more characters then a slash are in the string
    string(REGEX REPLACE "(.+)/$" "\\1" _working_dir ${_working_dir})

    # if the specified working directory is an absolute path
    if(IS_ABSOLUTE ${_working_dir})
      # check if specified working directory is part of the build directory
      cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${_working_dir} _wd_in_bd)
      if(NOT ${_wd_in_bd})
        message(FATAL_ERROR "WORKING_DIRECTORY has to be a directory inside the build directory!")
      endif()
      # overwrite default working dir for debug run target
      set(VID_DEBUG_RUN_TARGET_${_target_upper}_WORKING_DIR ${_working_dir})
    else()
      # overwrite default working dir for debug run target
      set(VID_DEBUG_RUN_TARGET_${_target_upper}_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/${_working_dir})
    endif()
  endif()

  # clear internal variables from previous calls of vid_debug_run_target()
  unset(_host_target)
  unset(_debug_target_outputs)
  unset(_host_executable)
  unset(_host_output_redirect)

  # set default value for target_has_host to FALSE
  set(_target_has_host FALSE)
  # check if HOST_APP is DEFINED
  if(DEFINED _ARG_HOST_APP)
    # set name of host target
    set(_host_target ${_ARG_HOST_APP})
    # check if host_target is a valid cmake target
    if(TARGET ${_host_target})
      set(_target_has_host TRUE)
      # get name of the host executable (normally target name, only for safety reasons)
      get_target_property(_host_target_name ${_host_target} NAME)
      # get directory where host target is build
      get_target_property(_host_target_dir ${_host_target} BINARY_DIR)
      # combine directory of host target build and host target name to full executable path
      set(_host_executable ${_host_target_dir}/${_host_target_name})
      # do additional preparation in the case the host application should be forked by the debugger
      if(${_ARG_FORK_HOST})
        # transfer the list with host arguments from macro call into a whitespace separated string
        string(REPLACE ";" " " _host_args_spaced "${_ARG_HOST_ARGS}")
        # create stripped argument for -host_app parameter
        string(STRIP "${_host_executable} ${_host_args_spaced}" VID_DEBUG_RUN_TARGET_HOST_APP)
        # set host stdout file
        set(VID_DEBUG_RUN_TARGET_HOST_STDOUT ${_target}.host.txt)
      endif()
    else()
      message(FATAL_ERROR "Defined host target ${_host_target} is not available")
    endif()
  endif()

  # check if host application for simulation is defined and HOST_ARGS are defined
  # raise warning to make aware that HOST_ARGS will be ignored
  if((NOT ${_target_has_host}) AND (DEFINED _ARG_HOST_ARGS))
    message(WARNING "HOST_ARGS is specified when calling macro vid_debug_run_target(), "
      "but no host application is defined via HOST_APP. "
      "HOST_ARGS will be ignored")
  endif()

  # prepare variables to be inserted into debugger configuration template
  _prepare_cfg_variables(${_target} debug ${_working_dir})

  # prepare debugger configuration for debugging
  set(_dbgcfg ${_target}.dbgcfg)

  # output: _FILE_CONTENT
  vid_cached_configure_file(${VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE} ${VID_DEBUG_RUN_TARGET_${_target_upper}_WORKING_DIR}/${_dbgcfg})

  # add generated dbgcfg to debug run targets inputs
  set(_debug_target_inputs ${VID_DEBUG_RUN_TARGET_${_target_upper}_WORKING_DIR}/${_dbgcfg})

  # parse the resolved content directly in memory (avoids writing to disk and re-reading the
  # unique generated file per target)
  # output: _SIMCFG_LIST
  vid_parse_simcfg_as_list(FROM_STRING "${_FILE_CONTENT}")

  # parse the generated dbgcfg for "-f" parameter
  # output: _SIMCFG_ARG
  vid_get_arg_from_simcfg_list(-f SIMCFG ${_SIMCFG_LIST} MULTIPLE)
  # iterate over found parameter files
  foreach(_arg ${_SIMCFG_ARG})
    # check if absolute path is defined/generated and the file exists
    if((IS_ABSOLUTE ${_arg}) AND (EXISTS ${_arg}))
      # append parameter file to run target inputs
      # this ensures cmake runs again without user interaction, if the file is changed
      list(APPEND _debug_target_inputs ${_arg})
    else()
      message(FATAL_ERROR "The parameter file found in the generated simcfg does not exists or has no absolute path: ${_arg} "
        "NOTE: Best practice is to define the path to the file as absolute path with the help of @VID_DEBUG_RUN_TARGET_SIMCFG_PATH@")
    endif()
  endforeach(_arg)

  # get simulator inputs from generated simcfg
  # iterate over simulator arguments that require input files
  foreach(_input_arg ${VIDSIM-DBG_INPUT_ARGS})
    # parse current arguments from simcfg
    # (multiple definitions will be parsed; filenames are always on position 0 for every argument)
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(${_input_arg} POS 0 SIMCFG ${_SIMCFG_LIST} MULTIPLE)
    # iterate over found files
    foreach(_file ${_SIMCFG_ARG})
      # append parsed argument to debug target inputs list
      list(APPEND _debug_target_inputs ${_file})
    endforeach(_file)
  endforeach(_input_arg)

  # get debugger outputs from generated dbgcfg
  # iterate over simulator arguments that generate output files
  foreach(_output_arg ${VIDSIM-DBG_OUTPUT_ARGS})
    # parse current arguments from simcfg
    # (multiple definitions will be parsed; filenames are always on position 0 for every argument)
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(${_output_arg} POS 0 SIMCFG ${_SIMCFG_LIST} MULTIPLE)
    # iterate over found files
    foreach(_file ${_SIMCFG_ARG})
      # check if provided output has an absolute path
      if(IS_ABSOLUTE ${_file})
        # append output with absolute path from _ARG_OUTPUTS to debug target outputs
        list(APPEND _debug_target_outputs ${_file})
      else()
        # if working directory is current binary directory
        if(${_working_dir} STREQUAL ".")
          # append parsed argument to debug target outputs list
          list(APPEND _debug_target_outputs ${_file})
        else()
          # append parsed argument to debug target outputs list with debug run dir path
          list(APPEND _debug_target_outputs ${_working_dir}/${_file})
        endif()
      endif()
    endforeach(_file)
  endforeach(_output_arg)

  # append INPUTS to debug target inputs
  # (only if defined otherwise nothing will be appended)
  list(APPEND _debug_target_inputs ${_ARG_INPUTS})
  # iterate over _ARG_OUTPUTS
  # (only if defined otherwise nothing will be appended)
  foreach(_output ${_ARG_OUTPUTS})
    # check if provided output has an absolute path
    if(IS_ABSOLUTE ${_output})
      # append output with absolute path from _ARG_OUTPUTS to debug target outputs
      list(APPEND _debug_target_outputs ${_output})
    else()
      # if working directory is current binary directory
      if(${_working_dir} STREQUAL ".")
        # append output from _ARG_OUTPUTS to debug target outputs
        list(APPEND _debug_target_outputs ${_output})
      else()
        # append output from _ARG_OUTPUTS to debug target outputs with debug run dir path
        list(APPEND _debug_target_outputs ${_working_dir}/${_output})
      endif()
    endif()
  endforeach(_output)
  # remove possible duplicates from debug target inputs and outputs lists
  list(REMOVE_DUPLICATES _debug_target_inputs)
  list(REMOVE_DUPLICATES _debug_target_outputs)

  # preparation for running a host application forked by debugger
  if(${_target_has_host} AND ${_ARG_FORK_HOST})
    # check if dbgcfg has parameter "-host_app" set
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(-host_app SIMCFG ${_SIMCFG_LIST})

    # check if "-host_app" was defined in dbgcfg
    if(DEFINED _SIMCFG_ARG)
      # get host application executable as first element of list
      # the host application executable must be always the first element after "-host_app"
      list(GET _SIMCFG_ARG 0 _host_app_executable)
      # check if host application executable has a absolute path
      if(NOT IS_ABSOLUTE ${_host_app_executable})
        # make host application executable path absolute
        get_filename_component(_host_app_executable ${_host_app_executable} ABSOLUTE)
      endif()
      # check if found host application executable matches the defined one
      if(NOT ${_host_executable} STREQUAL ${_host_app_executable})
        message(FATAL_ERROR "Host application executable found in the debugger configuration does not match the host application defined via HOST_APP")
      endif()
    else()
      # no "-host_app" parameter found in debugger configuration
      # create string to print "-host_app" vidsim-dbg argument
      set(_vidsim_dbg_extra_args " -host_app \"${VID_DEBUG_RUN_TARGET_HOST_APP}\"")
    endif()

    # check if dbgcfg has parameter "-host_app_stdout" set
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(-host_app_stdout SIMCFG ${_SIMCFG_LIST} IS_SET)
    # if "-host_app_stdout" is not set
    # and host application stdout should not be visible on stdout
    # in the case "-host_app_stdout" is set in the dbgcfg, the output will be added by default to cmake outputs
    if((NOT ${_SIMCFG_ARG}) AND (NOT ${VIDSDK_HOST_STDOUT}))
      # if working directory is current binary directory
      if(${_working_dir} STREQUAL ".")
        # add forked host application stdout file to debug target outputs
        list(APPEND _debug_target_outputs ${_target}.host.txt)
      else()
        # add forked host application stdout file to debug target outputs
        list(APPEND _debug_target_outputs ${_working_dir}/${_target}.host.txt)
      endif()
      # no "-host_app_stdout" parameter found in debugger configuration
      # create string to print "-host_app_stdout" vidsim-dbg argument
      set(_vidsim_dbg_extra_args "${_vidsim_dbg_extra_args} -host_app_stdout ${_target}.host.txt")
    # raise warning if "-host_stdout" parameter is set in dbgcfg
    # and host application stdout should be visible on stdout
    elseif(${_SIMCFG_ARG} AND ${VIDSDK_HOST_STDOUT})
      message(WARNING "CMake option VIDSDK_HOST_STDOUT is enabled, but the debugger configuration has \"-host_stdout\" set. "
        "Change debugger configuration template to see host application output on stdout: ${VID_DEBUG_RUN_TARGET_${_target_upper}_DBGCFG_TEMPLATE}")
    endif()

    # check if simcfg has parameter "-host_app_args" set and HOST_ARGS is specified in macro call
    # output: _SIMCFG_ARG
    vid_get_arg_from_simcfg_list(-host_app_args SIMCFG ${_SIMCFG_LIST} IS_SET)
    # warning can be silenced when calling macro with NO_WARN_HOST_ARGS
    if(${_SIMCFG_ARG} AND (DEFINED _ARG_HOST_ARGS) AND (NOT ${_ARG_NO_WARN_HOST_ARGS}))
      message(WARNING "Macro vid_debug_run_target() is called with HOST_ARGS and specified the debugger "
        "configuration has \"-host_app_args\" set. \"-host_app_args\" will be appended to the HOST_ARGS. "
        "This warning can be silenced by calling macro vid_debug_run_target() with option NO_WARN_HOST_ARGS")
    endif()
  endif()

  # get the cmake project name in lower case letters
  string(TOLOWER ${CMAKE_PROJECT_NAME} _project_name_lower)
  # set debugger configuration for the root of the cmake build directory
  set(_dbgcfg_root ${CMAKE_BINARY_DIR}/${_project_name_lower}.dbgcfg)
  # define name simulator/debugger config file needed by the lowlevel library on host side
  set(_simcfg vid_sim_config_link.cfg)

  # check if cmake should run the host application
  # (host application specified and no forking via debugger)
  if((DEFINED _ARG_HOST_APP) AND (NOT ${_ARG_FORK_HOST}))
    # set log file for host stdout
    set(_host_out_file ${_target}.host.txt)
    # check if host application stdout should be visible on stdout
    if(${VIDSDK_HOST_STDOUT})
      # define redirecting the output of host application to log file and stdout
      set(_host_output_redirect | tee ${_host_out_file})
    else()
      # define redirecting the output of host application to log file
      set(_host_output_redirect > ${_host_out_file})
    endif()
    # if working directory is current binary directory
    if(${_working_dir} STREQUAL ".")
      # add log file of host stdout file list of debug target outputs
      list(APPEND _debug_target_outputs ${_host_out_file})
    else()
      # add log file of host stdout file list of debug target outputs
      list(APPEND _debug_target_outputs ${_working_dir}/${_host_out_file})
    endif()
    # define custom target to copy generated debugger config with generic name to the root
    # of the cmake build directory
    add_custom_target(debug_${_target}_helper
      COMMAND ${CMAKE_COMMAND} -E copy ${_dbgcfg} ${_dbgcfg_root}
      COMMAND ${CMAKE_COMMAND} -E echo "Start vidsim-dbg with the following arguments: ${_dbgcfg_root}"
      DEPENDS ${_debug_target_inputs} ${_host_target}
      WORKING_DIRECTORY ${_working_dir}
      VERBATIM COMMENT "Copying videantis debugger configuration...")
    # define custom target to create a symlink for the dbgcfg for the lowlevel library
    # of the host and start host application
    add_custom_target(debug_${_target}
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${_dbgcfg} ${_simcfg}
      COMMAND $<TARGET_FILE:${_host_target}> ${_ARG_HOST_ARGS} ${_host_output_redirect}
      BYPRODUCTS ${_debug_target_outputs} ${_working_dir}/${_simcfg}
      DEPENDS debug_${_target}_helper ${_debug_target_inputs} ${_host_target}
      WORKING_DIRECTORY ${_working_dir}
      VERBATIM COMMENT "Starting host application in ${VID_DEBUG_RUN_TARGET_${_target_upper}_WORKING_DIR} ...")
  else()
    # check if the debugger should fork the host application
    if(${_ARG_FORK_HOST})
      # define custom command create a symlink for the dbgcfg for the lowlevel library
      # of the host
      add_custom_command(OUTPUT ${_working_dir}/${_simcfg}
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${_dbgcfg} ${_simcfg}
        DEPENDS ${_working_dir}/${_dbgcfg}
        WORKING_DIRECTORY ${_working_dir}
        COMMENT "Linking videantis debugger configuration for host application ...")
        # define custom target to copy generated debugger config with generic name to the
        # root of the cmake build directory
      add_custom_target(debug_${_target}
        COMMAND ${CMAKE_COMMAND} -E copy ${_dbgcfg} ${_dbgcfg_root}
        COMMAND ${CMAKE_COMMAND} -E echo "Start vidsim-dbg with the following arguments:${_vidsim_dbg_extra_args} ${_dbgcfg_root}"
        BYPRODUCTS ${_debug_target_outputs}
        DEPENDS ${_debug_target_inputs} ${_host_target} ${_working_dir}/${_simcfg}
        WORKING_DIRECTORY ${_working_dir}
        VERBATIM COMMENT "Copying videantis debugger configuration ...")
    else()
      # define custom target to copy generated debugger config with generic name to the
      # root of the cmake build directory
      add_custom_target(debug_${_target}
        COMMAND ${CMAKE_COMMAND} -E copy ${_dbgcfg} ${_dbgcfg_root}
        COMMAND ${CMAKE_COMMAND} -E echo "Start vidsim-dbg with the following arguments: ${_dbgcfg_root}"
        BYPRODUCTS ${_debug_target_outputs}
        DEPENDS ${_debug_target_inputs}
        WORKING_DIRECTORY ${_working_dir}
        VERBATIM COMMENT "Copying videantis debugger configuration ...")
    endif()
  endif()

  # add debugger config as additional file for clean up
  set_property(TARGET debug_${_target} APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${_dbgcfg_root})
endmacro(vid_debug_run_target)
