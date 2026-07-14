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
# FILENAME:    build.cmake
#
# DESCRIPTION: Support file for videantis SDK to build v-SP code
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

# when preparing to build a videantis v-SP target,
# register all lowlevel library components of all available library versions
# iterate over every available lowlevel library
foreach(_lllib ${VIDSDK_LOWLEVELLIBRARY})
  # get lowlevel library in upper case letters
  string(TOUPPER ${_lllib} _lllib_upper)
  # check if lowlevel library v-SP library dir is defined
  if(DEFINED VID_${_lllib_upper}_VSP_LIB_DIR)
    # get c files from lllib directory
    file(GLOB _vsp_lib_c_files LIST_DIRECTORIES FALSE "${VID_${_lllib_upper}_VSP_LIB_DIR}/*.c")
    # iterate over all lllib v-SP sources
    foreach(_source ${_vsp_lib_c_files})
      # register source file
      vid_register_file_vsp(${_source} ${_lllib})
    endforeach(_source)
    # create all upper case string of lowlevel library version
    string(TOUPPER ${_lllib} _lllib_upper)
    # write found and registered files as cached variable
    set(VIDSDK_${_lllib_upper}_VSP_C_FILES ${_vsp_lib_c_files} CACHE STRING "videantis SDK ${_lllib} c source files")
  endif()
endforeach(_lllib)

#[=======================================================================[.rst:
.. cmake:command:: vid_init_build_target_vsp

  Macro for initializing a videantis v-SP build target.
  Target gets defined and some default values get initialized.

  .. code-block:: cmake

    vid_init_build_target_vsp(name [VSP_BOOT_ADDRESS vsp_boot_address]
        [VSP_BOOT_ADDRESS_OFFSET vsp_boot_address_offset] [STACK_SIZE_IN_BITS stack_size_in_bits])

  Function parameters:

  ``name``
    name of the target to initialize
  ``VSP_BOOT_ADDRESS``
    v-SP boot address, defaults to 0x0
  ``VSP_BOOT_ADDRESS_OFFSET``
    v-SP boot address offset, defaults to 0x00010000
  ``STACK_SIZE_IN_BITS``
    size of stack in bits

  Exposed global properties:

  ``VID_TARGET_<TARGET_NAME>``
    name of build target (case sensitive)
  ``VID_TARGET_<TARGET_NAME>_TYPE``
    type of build target (VSP)
  ``VID_TARGET_<TARGET_NAME>_TARGET_CPU``
    cpu of the target (e.g. sp3.1)
  ``VID_TARGET_<TARGET_NAME>_TARGET_SOC``
    SoC of the target

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>``
    name of build target (variable only used to ensure other macro are called in the correct scope)
  ``VID_TARGET_<TARGET_NAME>_SP_VERSION``
    v-SP version depending on defined target cpu
  ``VID_TARGET_<TARGET_NAME>_SIZE_MEMORY_IMEM``
    size of IMEM in bytes (default: 32768)
  ``VID_TARGET_<TARGET_NAME>_SIZE_MEMORY_DMEM``
    size of DMEM in bytes (default: 32768)
  ``VID_TARGET_<TARGET_NAME>_SIZE_MEMORY_ICMEM``
    size of ICMEM in bytes (default: 4096, if available v-SP 3.3)
  ``VID_TARGET_<TARGET_NAME>_LLLIB_VSP_INCLUDE_DIR``
    lowlevel library include dir read from dependent pkgconfig of lllib
  ``VID_TARGET_<TARGET_NAME>_LLLIB_VSP_LIB_DIR``
    lowlevel library dir for assembler files
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_INCLUDES``
    list of include dirs for vspgcc
  ``VID_TARGET_<TARGET_NAME>_VSPASM_INCLUDES``
    list of include dirs for vspasm
  ``VID_TARGET_<TARGET_NAME>_VSP_LIB_ASM_FILES``
    assembler files of lllib (default: vid_vsp_boot_loader.asm, vid_vsp_lib.asm and vid_vsp_olm.asm)
  ``VID_TARGET_<TARGET_NAME>_VSP_LIB_C_FILES``
    c source files of lllib (default: vid_vsp_lib.c, vid_vsp_io.c, vid_lowlevel_vsp.c and vid_vsp_mbox.c)
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_PREPROCESSOR_FLAGS_DEFAULT``
    default flags for vspgcc preprocessor export (default: -E and -P)
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_PREPROCESSOR_POSTPROCESSING``
    commands for post processing of vspgcc preprocessor export (default: | tr '\\\;' '\\n')
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_OPTIMIZATION_LEVEL``
    vspgcc optimization level (default: -O2)
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_TARGET_FLAGS``
    list with cpu and soc flags for vspgcc
  ``VID_TARGET_<TARGET_NAME>_VSPASM_TARGET_FLAGS``
    list with cpu and soc flags for vspasm
  ``VID_TARGET_<TARGET_NAME>_VSPASM_STATISTIC_FLAGS``
    vspasm statistic flag (default: --statistic=MEMORYIMAGE,OVERLAY)
  ``VID_TARGET_<TARGET_NAME>_VSP_STACK_SIZE_IN_BITS``
    stack size in bits (if STACK_SIZE_IN_BITS is set when calling the marco)
  ``VID_TARGET_<TARGET_NAME>_VSPASM_OLM_BOOTADDR_FLAGS``
      vspasm olm boot address flags (default: --olm-initial-in-hardware-memory=0x0,
      --olm-external-offset=0x00010000)
  ``VID_TARGET_<TARGET_NAME>_VSPASM_HW_SOURCECODE_INFO_FLAGS``
    vspasm hw source code info flags (default: --enable-hw-sourcecode-info,
    --enable-absolute-sourcecode-info and
    --enable-comment-output)
  ``VID_TARGET_<TARGET_NAME>_VSPASM_OUT_FLAGS``
    vspasm out flags
    (default: --out-asm=<target_name>.out.asm,
    --out-ext-asm=<target_name>.ovl.out.asm
    --out-hw-asm=<target_name>.hw.asm,
    --out-code=<target_name>.imem,
    --out-ext-code=<target_name>.ovl.imem,
    --out-memory-map=<target_name>.nocodemap,
    --out-memory-dmem=<target_name>.dmem)
  ``VID_TARGET_<TARGET_NAME>_VSPASM_CUSTOM_FLAGS``
    variable to add custom flags for vspasm

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<target_name>``
    target name
#]=======================================================================]
macro(vid_init_build_target_vsp _name)
  cmake_parse_arguments(_ARG ""
    "VSP_BOOT_ADDRESS;VSP_BOOT_ADDRESS_OFFSET;STACK_SIZE_IN_BITS"
    "" ${ARGN})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_init_build_target_vsp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # set name with upper case
  string(TOUPPER ${_name} _name_upper)

  # check if name for this property is already set
  get_property(_target_property_is_set GLOBAL PROPERTY VID_TARGET_${_name_upper} SET)
  if(_target_property_is_set)
    message(FATAL_ERROR "Target ${_name} is already defined in videantis build context")
  endif()
  # set property for this target
  set_property(GLOBAL PROPERTY VID_TARGET_${_name_upper} ${_name})
  # set variable for this target (ensures all other functions will be called in the correct scope)
  set(VID_TARGET_${_name_upper} ${_name})

  # set property with core type for this target
  set_property(GLOBAL PROPERTY VID_TARGET_${_name_upper}_TYPE VSP)

  # get global defined lllib
  unset(_lllib)
  get_property(_lllib GLOBAL PROPERTY VID_TARGET_LLLIB)

  # check if lllib was set
  if(NOT DEFINED _lllib)
    message(FATAL_ERROR "No videantis lowlevel library is specified! "
      "Define a project wide lowlevel library with vid_define_system_vsp().")
  endif()

  # check if target cpu was set
  unset(_target_cpu)
  get_property(_target_cpu GLOBAL PROPERTY VID_TARGET_TARGET_CPU_SP)

  # set project wide target cpu or return error
  if(NOT DEFINED _target_cpu)
    message(FATAL_ERROR "No target cpu is specified! "
      "Define a project wide target cpu with vid_define_system_vsp().")
  endif()

  # set global property with target cpu for this build target
  set_property(GLOBAL PROPERTY VID_TARGET_${_name_upper}_TARGET_CPU ${_target_cpu})

  # get global defined target soc
  unset(_target_soc)
  get_property(_target_soc GLOBAL PROPERTY VID_TARGET_TARGET_SOC_SP)

  # check if target soc was set
  if(NOT DEFINED _target_soc)
    message(FATAL_ERROR "No target soc is specified! "
      "Define a project wide target soc with vid_define_system_vsp().")
  endif()

  # set global property with target soc for this build target
  set_property(GLOBAL PROPERTY VID_TARGET_${_name_upper}_TARGET_SOC ${_target_soc})

  # check if requested lowlevel library is available in the used SDK
  if(NOT ${_lllib} IN_LIST VIDSDK_LOWLEVELLIBRARY)
    message(FATAL_ERROR "${_lllib} is not available in ${VIDSDK}")
  endif()
  # create all upper case string of lowlevel library version
  string(TOUPPER ${_lllib} _lllib_upper)

  # check if v-SP lib dir is set (not defined means no v-SP lowlevel library included)
  if(NOT DEFINED VID_${_lllib_upper}_VSP_LIB_DIR)
    message(FATAL_ERROR "${_lllib} contains no v-SP lowlevel library!")
  endif()

  # set targets lowlevel lib
  set(VID_TARGET_${_name_upper}_LLLIB ${_lllib})

  # get v-SP version based on target_cpu
  string(REPLACE "sp" "" VID_TARGET_${_name_upper}_SP_VERSION ${_target_cpu})

  # set default sizes for instruction/data memories based on v-SP version
  set(VID_TARGET_${_name_upper}_SIZE_MEMORY_IMEM 32768)
  set(VID_TARGET_${_name_upper}_SIZE_MEMORY_DMEM 32768)
  if(${VID_TARGET_${_name_upper}_SP_VERSION} VERSION_EQUAL "3.3")
    set(VID_TARGET_${_name_upper}_SIZE_MEMORY_ICMEM 4096)
  endif()

  # check if lowlevel library include dirs are defined
  if(NOT DEFINED VID_${_lllib_upper}_HOST_INCLUDE_DIRS)
    message(FATAL_ERROR "Cmake variable VID_${_lllib_upper}_HOST_INCLUDE_DIRS is not defined.")
  endif()

  # check if lowlevel library v-SP library dir is defined
  if(NOT DEFINED VID_${_lllib_upper}_VSP_LIB_DIR)
    message(FATAL_ERROR "Cmake variable VID_${_lllib_upper}_VSP_LIB_DIR is not defined.")
  endif()

  # get lowlevel library include dir
  set(VID_TARGET_${_name_upper}_LLLIB_VSP_INCLUDE_DIR ${VID_${_lllib_upper}_HOST_INCLUDE_DIRS})

  # get lowlevel library v-SP lib dir
  set(VID_TARGET_${_name_upper}_LLLIB_VSP_LIB_DIR ${VID_${_lllib_upper}_VSP_LIB_DIR})

  # initialize vspgcc includes with v-SP include dir of lllib and lib dir of lllib
  set(VID_TARGET_${_name_upper}_VSPGCC_INCLUDES ${VID_TARGET_${_name_upper}_LLLIB_VSP_INCLUDE_DIR} ${VID_TARGET_${_name_upper}_LLLIB_VSP_LIB_DIR})

  # initialize vspasm includes with v-SP lib dir of lllib and current binary dir
  # (compiled *.c sources will be found in the current binary dir)
  set(VID_TARGET_${_name_upper}_VSPASM_INCLUDES ${VID_TARGET_${_name_upper}_LLLIB_VSP_LIB_DIR} ${CMAKE_CURRENT_BINARY_DIR})

  # set default library files for v-SP coming from the SDK
  set(VID_TARGET_${_name_upper}_VSP_LIB_ASM_FILES vid_vsp_boot_loader.asm vid_vsp_lib.asm vid_vsp_olm.asm)
  # iterate over found lowlevel library c files and save filename
  foreach(_file ${VIDSDK_${_lllib_upper}_VSP_C_FILES})
    # get filename without path
    get_filename_component(_filename ${_file} NAME)
    # add filename to list
    list(APPEND VID_TARGET_${_name_upper}_VSP_LIB_C_FILES ${_filename})
  endforeach(_file)

  # set default flags for performing a vspgcc preprocessor export
  set(VID_TARGET_${_name_upper}_VSPGCC_PREPROCESSOR_FLAGS_DEFAULT -E -P)

  # set default command for post processing a vspgcc preprocessor export
  set(VID_TARGET_${_name_upper}_VSPGCC_PREPROCESSOR_POSTPROCESSING | tr '\\\;' '\\n')

  # set default vspgcc optimization level
  set(VID_TARGET_${_name_upper}_VSPGCC_OPTIMIZATION_LEVEL "-O2")

  # create target compile flags based on target_cpu and target_soc
  set(VID_TARGET_${_name_upper}_VSPGCC_TARGET_FLAGS "-mcpu=${_target_cpu}" "-msoc=${_target_soc}")

  # create target assemble flags based on target_cpu and target_soc
  set(VID_TARGET_${_name_upper}_VSPASM_TARGET_FLAGS "--cpu=${_target_cpu}" "--soc=${_target_soc}")

  # set default vspasm statistic flags
  set(VID_TARGET_${_name_upper}_VSPASM_STATISTIC_FLAGS "--statistic=MEMORYIMAGE,OVERLAY")

  # define STACK_SIZE_IN_BITS when argument is given
  if(DEFINED _ARG_STACK_SIZE_IN_BITS)
    set(VID_TARGET_${_name_upper}_VSP_STACK_SIZE_IN_BITS ${_ARG_STACK_SIZE_IN_BITS})
  endif()

  # set default v-SP boot address
  set(_vsp_boot_address "0x0")
  # if v-SP boot address is defined when calling this macro overwrite default value
  if(DEFINED _ARG_VSP_BOOT_ADDRESS)
    set(_vsp_boot_address ${_ARG_VSP_BOOT_ADDRESS})
  endif()
  # set default v-SP boot address offset
  set(_vsp_boot_address_offset "0x00010000")
  # if v-SP boot address offset is defined when calling this macro overwrite default value
  if(DEFINED _ARG_VSP_BOOT_ADDRESS_OFFSET)
    set(_vsp_boot_address_offset ${_ARG_VSP_BOOT_ADDRESS_OFFSET})
  endif()
  # set default vspasm olm boot address flags
  set(VID_TARGET_${_name_upper}_VSPASM_OLM_BOOTADDR_FLAGS "--olm-initial-in-hardware-memory=${_vsp_boot_address}"
                                                          "--olm-external-offset=${_vsp_boot_address_offset}")

  # set default vspasm hw source code info flags
  set(VID_TARGET_${_name_upper}_VSPASM_HW_SOURCECODE_INFO_FLAGS "--enable-hw-sourcecode-info"
                                                                "--enable-absolute-sourcecode-info"
                                                                "--enable-comment-output")

  # set default vspasm out flags
  set(VID_TARGET_${_name_upper}_VSPASM_OUT_FLAGS "--out-asm=${_name}.out.asm"
                                                 "--out-ext-asm=${_name}.ovl.out.asm"
                                                 "--out-hw-asm=${_name}.hw.asm"
                                                 "--out-code=${_name}.imem"
                                                 "--out-ext-code=${_name}.ovl.imem"
                                                 "--out-memory-map=${_name}.nocodemap"
                                                 "--out-memory-dmem=${_name}.dmem")
endmacro(vid_init_build_target_vsp)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_sources_vsp

  Macro for adding source files to a videantis v-SP build target.
  This macro requires at least one IDENTIFIER to add source files
  or one source file added as local source file via ADD.
  If not all source files of one IDENTIFIER should be added,
  the files can be selected with FILES.

  .. code-block:: cmake

    vid_add_sources_vsp(target [IDENTIFIERS id1 [id2 ...]]
        [FILES file1 [file2 ...]] [ADD file1 [file2 ...]] [PRINT_AVAILABLE_SOURCES])

  Function parameters:

  ``target``
    name of an already defined build target
  ``IDENTIFIERS``
    list of identifiers for source files
  ``FILES``
    list of files from identifier

  .. note::

    Only one identifier is allow when FILES is set (:cmake:command:`vid_add_sources_vsp` can be called multiple times)

  ``ADD``
    list of files add as sources without registering as globally available file
  ``PRINT_AVAILABLE_SOURCES``
    print globally available sources

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VSPGCC_SOURCES``
    list of source files for vspgcc with full path
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_OUTPUTS``
    list of generated files of vspgcc (<target_name>_[<identifier>_]<source_filename_we>.asm)

  .. note::

    lists of vspgcc sources and outputs are mapped lists (same index for belonging files)

  ``VID_TARGET_<TARGET_NAME>_VSPASM_SOURCES``
    list of source files for vspasm with full path

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<target_name>``
    target name
  ``<identifier>``
    identifier if file was globally added with :cmake:command:`vid_register_file_vsp`
  ``<source_filename_we>``
    source file name without file extension
#]=======================================================================]
macro(vid_add_sources_vsp _target)
  cmake_parse_arguments(_ARG "PRINT_AVAILABLE_SOURCES" ""
    "IDENTIFIERS;FILES;ADD" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_add_sources_vsp() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VSP")
    message(FATAL_ERROR "vid_add_sources_vsp() needs to be called for a v-SP target")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_sources() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # get length of lists for IDENTIFIERS
  list(LENGTH _ARG_IDENTIFIERS _num_identifiers)

  # do error handling when calling this macro pending on input arguments
  # setting IDENTIFIERS is required when no ADD is specified
  if((${_num_identifiers} EQUAL 0) AND (NOT DEFINED _ARG_ADD))
    message(FATAL_ERROR "vid_add_sources_vsp() requires minimum one IDENTIFIER or one file via ADD")
  endif()
  # when sources are selected with FILES only one IDENTIFIER is allowed
  if((${_num_identifiers} GREATER 1) AND (DEFINED _ARG_FILES))
    message(FATAL_ERROR "vid_add_sources_vsp() can take only multiple IDENTIFIERs, when no FILES sources are specified")
  endif()
  if((DEFINED _ARG_FILES) AND (${_num_identifiers} EQUAL 0))
    message(FATAL_ERROR "vid_add_sources_vsp() needs one IDENTIFIER, when FILES sources are specified")
  endif()

  # unset print_available_sources from possible prior calls of vid_add_sources_vsp()
  unset(_print_available_sources)
  # check if PRINT_AVAILABLE_SOURCES is set
  if(${_ARG_PRINT_AVAILABLE_SOURCES})
    set(_print_available_sources PRINT_AVAILABLE_SOURCES)
  endif()

  # call internal macro to add sources
  _add_sources(${_target} vspgcc vspasm IDENTIFIERS ${_ARG_IDENTIFIERS} FILES ${_ARG_FILES} ADD ${_ARG_ADD} ${_print_available_sources})
endmacro(vid_add_sources_vsp)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_includes_vsp

  Macro for adding include directories to a videantis v-SP build target.
  This macro requires at least one IDENTIFIER to add include dirs
  or one directory added as local directory via ADD or ADD_VSPASM.
  If not all include dirs of one IDENTIFIER should be added,
  the directories can be selected with DIRECTORIES/DIRECTORIES_VSPASM.

  .. code-block:: cmake

    vid_add_includes_vsp(target [IDENTIFIERS id1 [id2 ...]]
        [DIRECTORIES dir1 [dir2 ...]] [DIRECTORIES_VSPASM dir1 [dir2 ...]]
        [ADD dir1 [dir2 ...]] [ADD_VSPASM dir1 [dir2 ...]] [PRINT_AVAILABLE_INCLUDES])

  Function parameters:

  ``target``
    name of an already defined build target
  ``IDENTIFIERS``
    list of identifiers for include dirs
  ``DIRECTORIES``
    list of dirs from the identifier (no full path)
  ``DIRECTORIES_VSPASM``
    list of dirs from the identifier (no full path)

  .. note::

    Only one identifier is allowed when DIRECTORIES/DIRECTORIES_VSPASM is set (:cmake:command:`vid_add_includes_vsp` can be called multiple times).
    If an include directory is registered as . (dot), the character . (dot) has to be used to select it.

  ``ADD``
    list of dirs add as includes without registering as globally available dir
  ``ADD_VSPASM``
    list of dirs add as includes for vspasm without registering as globally available dir
  ``PRINT_AVAILABLE_INCLUDES``
    print globally available includes

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VSPGCC_INCLUDES``
    list of include directories for vspgcc with full path
  ``VID_TARGET_<TARGET_NAME>_VSPASM_INCLUDES``
    list of include directories for vspasm with full path

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
#]=======================================================================]
macro(vid_add_includes_vsp _target)
  cmake_parse_arguments(_ARG "PRINT_AVAILABLE_INCLUDES" ""
    "IDENTIFIERS;DIRECTORIES;DIRECTORIES_VSPASM;ADD;ADD_VSPASM" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_add_includes_vsp() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VSP")
    message(FATAL_ERROR "vid_add_includes_vsp() needs to be called for a v-SP target")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_includes_vsp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # get length of lists for IDENTIFIERS
  list(LENGTH _ARG_IDENTIFIERS _num_identifiers)

  # do error handling when calling this macro pending on input arguments
  # setting IDENTIFIERS is required
  if((${_num_identifiers} EQUAL 0) AND (NOT DEFINED _ARG_ADD) AND (NOT DEFINED _ARG_ADD_VSPASM))
    message(FATAL_ERROR "vid_add_includes_vsp() requires minimum one IDENTIFIER or one dir via ADD/ADD_VSPASM")
  endif()
  # when paths are selected with DIRECTORIES only one IDENTIFIER is allowed
  if((${_num_identifiers} GREATER 1) AND ((DEFINED _ARG_DIRECTORIES) OR (DEFINED _ARG_DIRECTORIES_VSPASM)))
    message(FATAL_ERROR "vid_add_includes_vsp() can take only multiple IDENTIFIERS, when no DIRECTORIES/DIRECTORIES_VSPASM paths are specified")
  endif()
  if(((DEFINED _ARG_DIRECTORIES) OR (DEFINED _ARG_DIRECTORIES_VSPASM)) AND (${_num_identifiers} EQUAL 0))
    message(FATAL_ERROR "vid_add_includes_vsp() needs one IDENTIFIER, when DIRECTORIES/DIRECTORIES_VSPASM paths are specified")
  endif()

  # unset print_available_includes from possible prior calls of vid_add_includes_vsp()
  unset(_print_available_includes)
  # check if PRINT_AVAILABLE_INCLUDES is set
  if(${_ARG_PRINT_AVAILABLE_INCLUDES})
    set(_print_available_includes PRINT_AVAILABLE_INCLUDES)
  endif()

  # call internal macro to add includes
  _add_includes(${_target} vspgcc vspasm IDENTIFIERS ${_ARG_IDENTIFIERS} DIRECTORIES ${_ARG_DIRECTORIES}
    DIRECTORIES_VSPASM ${_ARG_DIRECTORIES_VSPASM} ADD ${_ARG_ADD} ADD_VSPASM ${_ARG_ADD_VSPASM} ${_print_available_includes})
endmacro(vid_add_includes_vsp)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_defines_vsp

  Macro for adding defines to a videantis v-SP build target.
  The define needs to be specified without "-D" in the beginning
  and is not allowed to contain whitespace.

  .. code-block:: cmake

    vid_add_defines_vsp(target [ALL define1 [define2 ...]] [VSPGCC define1 [define2 ...]]
        [VSPGCC_PREPROCESSOR define1 [define2 ...]] [VSPASM define1 [define2 ...]])

  Function parameters:

  ``target``
    name of an already defined build target
  ``ALL``
    add one or more defines to vspgcc and vspasm
  ``VSPGCC``
    add one or more defines to vspgcc
  ``VSPGCC_PREPROCESSOR``
    add one or more defines to vspgcc preprocessor export
  ``VSPASM``
    add one or more defines to vspasm

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VSPGCC_DEFINES``
    list of defines for vspgcc
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_PREPROCESSOR_DEFINES``
    list of defines for vspgcc preprocessor export
  ``VID_TARGET_<TARGET_NAME>_VSPASM_DEFINES``
    list of defines for vspasm

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
#]=======================================================================]
macro(vid_add_defines_vsp _target)
  cmake_parse_arguments(_ARG "" ""
    "ALL;VSPGCC;VSPGCC_PREPROCESSOR;VSPASM" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_add_defines_vsp() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VSP")
    message(FATAL_ERROR "vid_add_defines_vsp() needs to be called for a v-SP target")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_defines_vsp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # check if at least one argument (ALL, VSPGCC, VSPGCC_PREPROCESSOR or VSPASM) with value is present in the macro call
  if((NOT DEFINED _ARG_ALL) AND (NOT DEFINED _ARG_VSPGCC) AND (NOT DEFINED _ARG_VSPGCC_PREPROCESSOR) AND (NOT DEFINED _ARG_VSPASM))
    message(FATAL_ERROR "Argument (ALL, VSPGCC VSPGCC_PREPROCESSOR or VSPASM) and/or value is missing")
  endif()

  # append defines to vspgcc and vspasm define lists
  list(APPEND VID_TARGET_${_target_upper}_VSPGCC_DEFINES ${_ARG_ALL})
  list(APPEND VID_TARGET_${_target_upper}_VSPGCC_PREPROCESSOR_DEFINES ${_ARG_ALL})
  list(APPEND VID_TARGET_${_target_upper}_VSPASM_DEFINES ${_ARG_ALL})

  # append defines to vspgcc define list
  list(APPEND VID_TARGET_${_target_upper}_VSPGCC_DEFINES ${_ARG_VSPGCC})

  # append defines to vspgcc preprocessor define list
  list(APPEND VID_TARGET_${_target_upper}_VSPGCC_PREPROCESSOR_DEFINES ${_ARG_VSPGCC_PREPROCESSOR})

  # append defines to vspasm define list
  list(APPEND VID_TARGET_${_target_upper}_VSPASM_DEFINES ${_ARG_VSPASM})
endmacro(vid_add_defines_vsp)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_compile_flags_vsp

  Macro for adding vspgcc flags to a videantis v-SP build target.
  The flags should contain no whitespace.

  .. code-block:: cmake

    vid_add_compile_flags_vsp(target [FLAGS flag1 [flag2 ...]]
        [FLAGS_PREPROCESSOR flag1 [flag2 ...]])

  Function parameters:

  ``target``
    name of an already defined build target
  ``FLAGS``
    add one or more flags for vspgcc
  ``FLAGS_PREPROCESSOR``
    add one or more flags for vspgcc preprocessor export

  Note: FLAGS or/and FLAGS_PREPROCESSOR needs to be specified

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VSPGCC_FLAGS``
    list of flags for vspgcc
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_PREPROCESSOR_FLAGS``
    list of flags for vspgcc preprocessor export

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
#]=======================================================================]
macro(vid_add_compile_flags_vsp _target)
  cmake_parse_arguments(_ARG "" ""
    "FLAGS;FLAGS_PREPROCESSOR" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_add_compile_flags_vsp() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VSP")
    message(FATAL_ERROR "vid_add_compile_flags_vsp() needs to be called for a v-SP target")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_compile_flags_vsp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # check if FLAGS is present in the macro call
  if((NOT DEFINED _ARG_FLAGS) AND (NOT DEFINED _ARG_FLAGS_PREPROCESSOR))
    message(FATAL_ERROR "Argument (FLAGS or FLAGS_PREPROCESSOR) and/or value is missing")
  endif()

  # append flags to vspgcc flag list
  list(APPEND VID_TARGET_${_target_upper}_VSPGCC_FLAGS ${_ARG_FLAGS})

  # append flags to vspgcc preprocessor flag list
  list(APPEND VID_TARGET_${_target_upper}_VSPGCC_PREPROCESSOR_FLAGS ${_ARG_FLAGS_PREPROCESSOR})
endmacro(vid_add_compile_flags_vsp)

#[=======================================================================[.rst:
.. cmake:command:: vid_build_target_vsp

  Macro for building a videantis v-SP build target.

  The cmake module defines for every build target a define for assembler
  and compiler with the target cpu. E.g. v-SP 3.2 target cpu creates the
  define `__sp302__`.

  .. code-block:: cmake

    vid_build_target_vsp(target [DEPENDENCIES dependency1 [dependency2 ...]]
        [WORKING_DIRECTORY dir] [ADD_INCLUDES_PREPROCESSOR] [BIN2H] [NO_TARGET] [NO_ALL])

  Function parameters:

  ``target``
    name of an already defined build target
  ``DEPENDENCIES``
    add one or more cmake dependencies
  ``WORKING_DIRECTORY``
    define working directory (default: current cmake binary directory)
  ``ADD_INCLUDES_PREPROCESSOR``
    add all vspgcc includes when performing a vspgcc preprocessor export
  ``BIN2H``
    create header file with binary code
  ``NO_TARGET``
    create no cmake targets
  ``NO_ALL``
    build target won't be added to ALL target of the build

  Exposed global variables:

  ``VID_TARGET_<TARGET_NAME>_WORKING_DIR``
    full path of the working directory of the build target
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_SOURCES``
    list of vspgcc sources of the build target
  ``VID_TARGET_<TARGET_NAME>_VSPASM_SOURCES``
    list of vspasm sources of the build target
  ``VID_TARGET_<TARGET_NAME>_SOURCES``
    combined list of all sources of the build target
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_OUTPUTS``
    list of vspgcc outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_VSPASM_OUTPUTS``
    list of vspasm outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_BINARY_OUTPUTS``
    list of binary outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_BIN2H_OUTPUTS``
    list of bin2h outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_OUTPUTS``
    combined list of all outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_VSPGCC_COMMANDS``
    list of vspgcc commands of the build target
  ``VID_TARGET_<TARGET_NAME>_VSPASM_COMMANDS``
    list of vspasm commands of the build target
  ``VID_TARGET_<TARGET_NAME>_BINARY_COMMANDS``
    list of binary commands of the build target
  ``VID_TARGET_<TARGET_NAME>_BIN2H_COMMANDS``
    list of bin2h commands of the build target
  ``VID_TARGET_<TARGET_NAME>_COMMANDS``
    combined list of all commands of the build target

  Exposed global properties:

  ``VID_TARGET_<TARGET_NAME>_BUILD``
    set to true when :cmake:command:`vid_build_target_vsp` is called
  ``VID_TARGET_<TARGET_NAME>_BIN_FILE``
    generated bin file (full path) for target
  ``VID_TARGET_<TARGET_NAME>_MAP_FILE``
    generated map file (full path) for target
  ``VID_TARGET_<TARGET_NAME>_BIN2H_FILE``
    generated bin2h file (full path) for target

  Exposed cmake targets:

  ``compile_<vspgcc_output_name>``
    compile only one source file (for \*.c files)
  ``preprocess_<vspgcc_output_name>``
    preprocess only one header file (for \*.h files)
  ``build_<target_name>``
    build target

  Placeholder:

  ``<target_name>``
    target name
  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<vspgcc_output_name>``
    <target_name>_[<identifier>_]<source_filename_we>
  ``<identifier>``
    identifier if file was globally added with :cmake:command:`vid_register_file_vsp`
  ``<source_filename_we>``
    vspgcc source file name without file extension
#]=======================================================================]
macro(vid_build_target_vsp _target)
  cmake_parse_arguments(_ARG "ADD_INCLUDES_PREPROCESSOR;BIN2H;NO_TARGET;NO_ALL" "WORKING_DIRECTORY"
    "DEPENDENCIES" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if vid_build_target_vsp() is already called for this target
  get_property(_target_build GLOBAL PROPERTY VID_TARGET_${_target_upper}_BUILD)
  if(DEFINED _target_build)
    message(FATAL_ERROR "vid_build_target_vsp() has been already called for target ${_target}")
  endif()
  # set this target as build, if vid_build_target_vsp() was not called before
  set_property(GLOBAL PROPERTY VID_TARGET_${_target_upper}_BUILD TRUE)

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_build_target_vsp() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VSP")
    message(FATAL_ERROR "vid_build_target_vsp() needs to be called for a v-SP target")
  endif()

  # check if bin2h executable was found when bin2h should be used
  if(${_ARG_BIN2H} AND (${BIN2H_EXECUTABLE} STREQUAL "BIN2H_EXECUTABLE-NOTFOUND"))
    message(FATAL_ERROR "Cannot create a header file with binary code inside: bin2h executable missing")
  endif()

  # raise warning if NO_TARGET and NO_ALL are defined together
  if(${_ARG_NO_TARGET} AND ${_ARG_NO_ALL})
    message(WARNING "NO_TARGET is defined for build target ${_target}! NO_ALL will have not effect")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_build_target_vsp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # set working dir for build target
  set(VID_TARGET_${_target_upper}_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR})
  # set default working directory (relative path of CMAKE_CURRENT_BINARY_DIR)
  set(_working_dir .)
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
      # overwrite default working dir for build target
      set(VID_TARGET_${_target_upper}_WORKING_DIR ${_working_dir})
    else()
      # overwrite default working dir for build target
      set(VID_TARGET_${_target_upper}_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/${_working_dir})
    endif()
  endif()

  # clear internal variables from previous calls of vid_build_target_vsp()
  unset(_vspgcc_include_flags)
  unset(_vspgcc_define_flags)
  unset(_vspasm_include_flags)
  unset(_vspasm_define_flags)
  unset(_vspasm_out_files)
  unset(_vspasm_mandatory_out_files)
  unset(_bin2h_file)
  unset(_lllib_c_files)
  unset(_lllib_c_files_wo_path)
  unset(_vsp_lib_asm_files)

  # create lowlevel library name in upper case letters
  string(TOUPPER ${VID_TARGET_${_name_upper}_LLLIB} _lllib_upper)
  # iterate overall c files of lowlevel library for v-SP
  foreach(_file ${VIDSDK_${_lllib_upper}_VSP_C_FILES})
    # get filename of current lowlevel library c file
    get_filename_component(_filename ${_file} NAME)
    # check if filename is still in list (user could update this list after initialization)
    if(${_filename} IN_LIST VID_TARGET_${_name_upper}_VSP_LIB_C_FILES)
      # append file to lowlevel library c files lists
      list(APPEND _lllib_c_files ${_file})
      list(APPEND _lllib_c_files_wo_path ${_filename})
    endif()
  endforeach(_file)
  # iterate over vspgcc sources
  foreach(_source ${VID_TARGET_${_target_upper}_VSPGCC_SOURCES})
    # check if file is in list of lowlevel library c files
    if(${_source} IN_LIST _lllib_c_files)
      # get filename of current source file
      get_filename_component(_source_wo_path ${_source} NAME)
      # remove item from lowlevel library c files lists
      list(REMOVE_ITEM _lllib_c_files ${_source})
      list(REMOVE_ITEM _lllib_c_files_wo_path ${_source_wo_path})
    endif()
  endforeach(_source)

  # get number of lowlevel lib c files to add
  list(LENGTH _lllib_c_files _lllib_c_files_length)
  # add files if at least one source file remained in list of lowlevel library c files
  if(${_lllib_c_files_length} GREATER "0")
    vid_add_sources_vsp(${_target} IDENTIFIERS ${VID_TARGET_${_target_upper}_LLLIB} FILES ${_lllib_c_files_wo_path})
  endif()

  # get number of list elements for vspgcc sources and outputs
  list(LENGTH VID_TARGET_${_target_upper}_VSPGCC_SOURCES _num_vspgcc_sources)
  list(LENGTH VID_TARGET_${_target_upper}_VSPGCC_OUTPUTS _num_vspgcc_outputs)
  # check if the number of vspgcc sources and outputs match
  if(NOT ${_num_vspgcc_sources} EQUAL ${_num_vspgcc_outputs})
    message(FATAL_ERROR "Number of sources and outputs for vspgcc do not match, something went wrong while setting up the target!")
  endif()

  # remove possible duplicates in lists of vspgcc sources and duplicates
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VSPGCC_SOURCES)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VSPGCC_OUTPUTS)
  # get number of list elements for vspgcc sources and outputs after removing possible duplicates
  list(LENGTH VID_TARGET_${_target_upper}_VSPGCC_SOURCES _num_vspgcc_sources)
  list(LENGTH VID_TARGET_${_target_upper}_VSPGCC_OUTPUTS _num_vspgcc_outputs)
  # check if the number of vspgcc sources and outputs match after removing possible duplicates
  if(NOT ${_num_vspgcc_sources} EQUAL ${_num_vspgcc_outputs})
    message(FATAL_ERROR "Number of sources and outputs for vspgcc do not match after removing duplicates. "
      "It could be possible that a file was added twice. Globally and locally or with different identifiers")
  endif()

  # remove possible duplicates from vspasm sources and includes of vspgcc and vspasm
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VSPASM_SOURCES)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VSPGCC_INCLUDES)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VSPASM_INCLUDES)

  # create target cpu define
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _ ${VID_TARGET_${_target_upper}_SP_VERSION})
  set(_sp_version_major ${CMAKE_MATCH_1})
  set(_sp_version_minor ${CMAKE_MATCH_2})
  if(${_sp_version_minor} LESS 10)
    set(_sp_version_minor 0${_sp_version_minor})
  endif()
  set(_target_cpu_define __core_version__=${_sp_version_major}${_sp_version_minor})
  # add target cpu as define
  list(APPEND VID_TARGET_${_target_upper}_VSPGCC_DEFINES ${_target_cpu_define})
  list(APPEND VID_TARGET_${_target_upper}_VSPASM_DEFINES ${_target_cpu_define})

  # if llvm stack is defined, append correct vspasm define to defines list
  if(DEFINED VID_TARGET_${_target_upper}_VSP_STACK_SIZE_IN_BITS)
    list(APPEND VID_TARGET_${_target_upper}_VSPASM_DEFINES STACK_SIZE_IN_BITS=${VID_TARGET_${_target_upper}_VSP_STACK_SIZE_IN_BITS})
  endif()

  # remove possible duplicates from vspgcc flags and defines of vspgcc and vspasm
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VSPGCC_FLAGS)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VSPGCC_DEFINES)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VSPASM_DEFINES)

  # iterate over vspgcc includes and add -I
  foreach(_vspgcc_include ${VID_TARGET_${_target_upper}_VSPGCC_INCLUDES})
    list(APPEND _vspgcc_include_flags "-I${_vspgcc_include}")
  endforeach(_vspgcc_include)

  # iterate over vspgcc defines and add -D
  foreach(_vspgcc_define ${VID_TARGET_${_target_upper}_VSPGCC_DEFINES})
    list(APPEND _vspgcc_define_flags "-D${_vspgcc_define}")
  endforeach(_vspgcc_define)

  # create a combined list of all vspgcc flags, includes and defines
  set(_vspgcc_flags ${VID_TARGET_${_target_upper}_VSPGCC_FLAGS}
                    ${VID_TARGET_${_target_upper}_VSPGCC_TARGET_FLAGS}
                    ${VID_TARGET_${_target_upper}_VSPGCC_OPTIMIZATION_LEVEL}
                    ${_vspgcc_include_flags}
                    ${_vspgcc_define_flags})

  # iterate over vspgcc defines and add -D
  foreach(_vspgcc_preprocessor_define ${VID_TARGET_${_target_upper}_VSPGCC_PREPROCESSOR_DEFINES})
    list(APPEND _vspgcc_pre_processor_define_flags "-D${_vspgcc_preprocessor_define}")
  endforeach(_vspgcc_preprocessor_define)

  # create a combined list of all vspgcc flags, includes and defines
  set(_vspgcc_preprocessor_flags ${VID_TARGET_${_target_upper}_VSPGCC_PREPROCESSOR_FLAGS_DEFAULT}
                                 ${VID_TARGET_${_target_upper}_VSPGCC_PREPROCESSOR_FLAGS}
                                 ${_vspgcc_pre_processor_define_flags})

  # check if vspgcc preprocessor export should get all vspgcc includes as flags
  if(${_ARG_ADD_INCLUDES_PREPROCESSOR})
    # append vspgcc include flags to list of vspgcc preprocessor flags
    list(APPEND _vspgcc_preprocessor_flags ${_vspgcc_include_flags})
  endif()

  # iterate over all vspgcc sources to create custom commands to compile every source
  foreach(_vspgcc_source _vspgcc_output IN ZIP_LISTS VID_TARGET_${_target_upper}_VSPGCC_SOURCES VID_TARGET_${_target_upper}_VSPGCC_OUTPUTS)
    # get output name without extension
    get_filename_component(_output_filename ${_vspgcc_output} NAME_WLE)

    # get source file filename
    get_filename_component(_source_filename ${_vspgcc_source} NAME)

    # extract file extension in upper case letters of vspgcc source file
    string(REGEX MATCH "\\.([a-zA-Z]+)$" _ ${_vspgcc_source})
    string(TOUPPER ${CMAKE_MATCH_1} _file_ext)

    # get length of vspgcc commands list
    list(LENGTH VID_TARGET_${_target_upper}_VSPGCC_COMMANDS _vspgcc_cmd_length)
    # if list of vspgcc commands contains something add && to list
    if(${_vspgcc_cmd_length} GREATER "0")
      list(APPEND VID_TARGET_${_target_upper}_VSPGCC_COMMANDS &&)
    endif()

    if(${_file_ext} STREQUAL "H")
      # add current vspgcc command to vspgcc commands list
      list(APPEND VID_TARGET_${_target_upper}_VSPGCC_COMMANDS ${VSPGCC_EXECUTABLE} ${_vspgcc_preprocessor_flags} ${_vspgcc_source} ${VID_TARGET_${_target_upper}_VSPGCC_PREPROCESSOR_POSTPROCESSING} > ${_vspgcc_output})

      # check if a target should be created
      if(NOT ${_ARG_NO_TARGET})
        # preprocess source with vspgcc
        add_custom_command(OUTPUT ${_working_dir}/${_vspgcc_output}
          COMMAND ${VSPGCC_EXECUTABLE} ${_vspgcc_preprocessor_flags} ${_vspgcc_source} ${VID_TARGET_${_target_upper}_VSPGCC_PREPROCESSOR_POSTPROCESSING} > ${_vspgcc_output}
          DEPENDS ${_vspgcc_source}
          WORKING_DIRECTORY ${_working_dir}
          VERBATIM COMMENT "Preprocessing ${_source_filename} to ${_vspgcc_output} ...")

        # add custom target for current output
        # this enables to run vspgcc only for one source (developing or debugging process)
        add_custom_target(preprocess_${_output_filename}
          DEPENDS ${_working_dir}/${_vspgcc_output})
      endif()
    else()
      # add current vspgcc command to vspgcc commands list
      list(APPEND VID_TARGET_${_target_upper}_VSPGCC_COMMANDS ${VSPGCC_EXECUTABLE} ${_vspgcc_flags} -o ${_vspgcc_output} ${_vspgcc_source})

      # check if a target should be created
      if(NOT ${_ARG_NO_TARGET})
        # compile source with vspgcc
        add_custom_command(OUTPUT ${_working_dir}/${_vspgcc_output} ${_working_dir}/${_vspgcc_output}.d
          COMMAND ${VSPGCC_EXECUTABLE} ${_vspgcc_flags} -MD ${_vspgcc_output}.d -MT ${_vspgcc_output} -o ${_vspgcc_output} ${_vspgcc_source}
          DEPENDS ${_vspgcc_source}
          DEPFILE ${_working_dir}/${_vspgcc_output}.d
          WORKING_DIRECTORY ${_working_dir}
          VERBATIM COMMENT "Compiling ${_source_filename} to ${_vspgcc_output} ...")

        # add custom target for current output
        # this enables to run vspgcc only for one source (developing or debugging process)
        add_custom_target(compile_${_output_filename}
          DEPENDS ${_working_dir}/${_vspgcc_output})
      endif()
    endif()
  endforeach(_vspgcc_source)

  # iterate over vspasm includes and add --include-dir=
  foreach(_vspasm_include ${VID_TARGET_${_target_upper}_VSPASM_INCLUDES})
    list(APPEND _vspasm_include_flags "--include-dir=${_vspasm_include}")
  endforeach(_vspasm_include)

  # iterate over vspasm defines and add -D
  foreach(_vspasm_define ${VID_TARGET_${_target_upper}_VSPASM_DEFINES})
    list(APPEND _vspasm_define_flags "-D${_vspasm_define}")
  endforeach(_vspasm_define)

  # set --size-memory-imem and --size-memory-dmem flags depending on the defined IMEM/DMEM size
  set(_vspasm_memory_flags "--size-memory-imem=${VID_TARGET_${_target_upper}_SIZE_MEMORY_IMEM}"
                           "--size-memory-dmem=${VID_TARGET_${_target_upper}_SIZE_MEMORY_DMEM}")
  # if sp version is equal 3.3 append icmem memory size flag
  if(${VID_TARGET_${_target_upper}_SP_VERSION} VERSION_EQUAL "3.3")
    list(APPEND _vspasm_memory_flags "--size-memory-icmem=${VID_TARGET_${_target_upper}_SIZE_MEMORY_ICMEM}")
  endif()

  # iterate over all vspasm out flags
  foreach(_vspasm_out_flag ${VID_TARGET_${_target_upper}_VSPASM_OUT_FLAGS})
    # strip flag to ensure no whitespace are at beginning or end of the flag
    string(STRIP ${_vspasm_out_flag} _vspasm_out_flag)
    # extract output filename from the flag
    string(REGEX MATCH "=([a-zA-Z0-9_\\.\\-]+)$" _ ${_vspasm_out_flag} )
    set(_vspasm_out_file ${CMAKE_MATCH_1})

    # remove extension from output filename
    get_filename_component(_vspasm_out_filename ${_vspasm_out_file} NAME_WE)
    # check that output filename without extension is the target (naming convention for easier processing)
    if(NOT ${_vspasm_out_filename} STREQUAL ${_target})
      message(FATAL_ERROR "vspasm output file (${_vspasm_out_file}) does not apply to naming convention: <target>.extension")
    endif()

    # append output file to list of output files
    list(APPEND _vspasm_out_files ${_vspasm_out_file})
  endforeach(_vspasm_out_flag)

  # define mandatory vspasm output files and append them to a list
  set(_vspasm_memory_map_file ${_target}.nocodemap)
  list(APPEND _vspasm_mandatory_out_files ${_vspasm_memory_map_file})
  set(_vspasm_hw_asm_file ${_target}.hw.asm)
  list(APPEND _vspasm_mandatory_out_files ${_vspasm_hw_asm_file})
  set(_vspasm_code_file ${_target}.imem)
  list(APPEND _vspasm_mandatory_out_files ${_vspasm_code_file})
  set(_vspasm_memory_code_ext_file ${_target}.ovl.imem)
  list(APPEND _vspasm_mandatory_out_files ${_vspasm_memory_code_ext_file})
  set(_vspasm_memory_dmem_file ${_target}.dmem)
  list(APPEND _vspasm_mandatory_out_files ${_vspasm_memory_dmem_file})

  # check if all mandatory output files are in the list of vspasm output files
  foreach(_vspasm_mandatory_out_file ${_vspasm_mandatory_out_files})
    if(NOT ${_vspasm_mandatory_out_file} IN_LIST _vspasm_out_files)
      message(FATAL_ERROR "File ${_vspasm_mandatory_out_file} is not in the list of vspasm outputs. "
        "Check VID_TARGET_${_target_upper}_VSPASM_OUT_FLAGS cmake variable!")
    endif()
  endforeach(_vspasm_mandatory_out_file)

  # create a combined list of all vspasm flags, includes and defines
  set(_vspasm_flags ${VID_TARGET_${_target_upper}_VSPASM_TARGET_FLAGS}
                    ${VID_TARGET_${_target_upper}_VSPASM_STATISTIC_FLAGS}
                    ${VID_TARGET_${_target_upper}_VSPASM_OLM_BOOTADDR_FLAGS}
                    ${VID_TARGET_${_target_upper}_VSPASM_HW_SOURCECODE_INFO_FLAGS}
                    ${_vspasm_include_flags}
                    ${_vspasm_define_flags}
                    ${_vspasm_memory_flags}
                    ${VID_TARGET_${_target_upper}_VSPASM_OUT_FLAGS}
                    ${VID_TARGET_${_target_upper}_VSPASM_CUSTOM_FLAGS})

  # iterate over all lowlevel library files specified
  foreach(_vsp_lib_asm_file ${VID_TARGET_${_target_upper}_VSP_LIB_ASM_FILES})
    # generate full path for lllib file based on LLLIB v-SP lib dir (defined in lllib cmake file)
    string(JOIN "/" _vsp_lib_asm_file_full_path ${VID_TARGET_${_target_upper}_LLLIB_VSP_LIB_DIR} ${_vsp_lib_asm_file})

    # check if file exists
    if(NOT EXISTS ${_vsp_lib_asm_file_full_path})
      message(FATAL_ERROR "${_vsp_lib_asm_file} does not exists in ${VID_TARGET_${_target_upper}_LLLIB_VSP_LIB_DIR}")
    endif()

    # append file to list with full path lllib files for dependencies of assembling step
    list(APPEND _vsp_lib_asm_files ${_vsp_lib_asm_file_full_path})
  endforeach(_vsp_lib_asm_file)

  # create a combined list of all vspasm sources
  # (assembler libraries, compiled c-code and assembler sources)
  set(_vspasm_sources ${VID_TARGET_${_target_upper}_VSP_LIB_ASM_FILES}
                      ${VID_TARGET_${_target_upper}_VSPGCC_OUTPUTS}
                      ${VID_TARGET_${_target_upper}_VSPASM_SOURCES})

  # check if additional sources besides the lllib are defined, if not raise error
  if((${_lllib_c_files_length} EQUAL ${_num_vspgcc_sources})
      AND (NOT DEFINED VID_TARGET_${_target_upper}_VSPASM_SOURCES))
    message(FATAL_ERROR "Target ${_target} has no sources defined. Use vid_add_sources_vsp() to add sources")
  endif()

  # append full path lowlevel library assembler files to vspasm sources list
  list(APPEND VID_TARGET_${_target_upper}_VSPASM_SOURCES ${_vsp_lib_asm_files})
  # set list with vspasm outputs
  set(VID_TARGET_${_target_upper}_VSPASM_OUTPUTS ${_vspasm_out_files})
  # set list with vspasm commands
  set(VID_TARGET_${_target_upper}_VSPASM_COMMANDS ${VSPASM_EXECUTABLE} ${_vspasm_flags} ${_vspasm_sources})

  # adapt path of vspgcc outputs, vspasm out files and vspasm mandatory out files,
  # if working directory is not current binary directory
  if(NOT ${_working_dir} STREQUAL ".")
    list(TRANSFORM VID_TARGET_${_target_upper}_VSPGCC_OUTPUTS PREPEND ${_working_dir}/)
    list(TRANSFORM _vspasm_out_files PREPEND ${_working_dir}/)
    list(TRANSFORM _vspasm_mandatory_out_files PREPEND ${_working_dir}/)
  endif()

  # check if a target should be created
  if(NOT ${_ARG_NO_TARGET})
    # assemble sources with vspasm
    add_custom_command(OUTPUT ${_vspasm_out_files}
      COMMAND ${VSPASM_EXECUTABLE} ${_vspasm_flags} ${_vspasm_sources}
      DEPENDS ${VID_TARGET_${_target_upper}_VSPGCC_OUTPUTS} ${VID_TARGET_${_target_upper}_VSPASM_SOURCES}
      WORKING_DIRECTORY ${_working_dir}
      VERBATIM COMMENT "Assembling ${_target} ...")
  endif()

  # define build target outputs
  set(_bin_file ${_target}.bin)
  set(_map_file ${_target}.map)

  # set list with binary outputs
  set(VID_TARGET_${_target_upper}_BINARY_OUTPUTS ${_map_file} ${_bin_file})
  # set list with binary commands
  set(VID_TARGET_${_target_upper}_BINARY_COMMANDS cat ${_vspasm_memory_map_file} ${_vspasm_hw_asm_file} > ${_map_file}
                                                  && cat ${_vspasm_code_file} ${_vspasm_memory_dmem_file} ${_vspasm_memory_code_ext_file} > ${_bin_file})

  # check if a target should be created
  if(NOT ${_ARG_NO_TARGET})
    # concatenate vspasm output files to map and bin file
    add_custom_command(OUTPUT ${_working_dir}/${_map_file} ${_working_dir}/${_bin_file}
      COMMAND cat ${_vspasm_memory_map_file} ${_vspasm_hw_asm_file} > ${_map_file}
      COMMAND cat ${_vspasm_code_file} ${_vspasm_memory_dmem_file} ${_vspasm_memory_code_ext_file} > ${_bin_file}
      DEPENDS ${_vspasm_mandatory_out_files}
      WORKING_DIRECTORY ${_working_dir}
      COMMENT "Generating map and bin file for ${_target} ...")
  endif()

  # check if a header file with binary code should be created
  if(${_ARG_BIN2H})
    # define bin2h output file
    set(_bin2h_file ${_target}.h)

    # set list with bin2h outputs
    set(VID_TARGET_${_target_upper}_BIN2H_OUTPUTS ${_bin2h_file})
    # set list with bin2h commands
    set(VID_TARGET_${_target_upper}_BIN2H_COMMANDS ${BIN2H_EXECUTABLE} ${_bin_file} ${_target} ${_bin2h_file})

    # check if a target should be created
    if(NOT ${_ARG_NO_TARGET})
      # add custom command to create binary header file
      add_custom_command(OUTPUT ${_working_dir}/${_bin2h_file}
        COMMAND ${BIN2H_EXECUTABLE} ${_bin_file} ${_target} ${_bin2h_file}
        DEPENDS ${_working_dir}/${_bin_file}
        WORKING_DIRECTORY ${_working_dir}
        COMMENT "Creating header file ${_bin2h_file} with binary code of ${_bin_file} ...")

      # add custom target for creating the binary header file
      # NOTE: this target will be used to ensure the binary header is build before the host is build
      add_custom_target(binary_header_${_target}
        DEPENDS ${_working_dir}/${_bin2h_file})
    endif()

    # set property for bin2h file of this target
    set_property(GLOBAL PROPERTY VID_TARGET_${_target_upper}_BIN2H_INC_DIR ${VID_TARGET_${_target_upper}_WORKING_DIR})
  endif()

  # check if a target should be created
  if(NOT ${_ARG_NO_TARGET})
    # unset all variable
    unset(_all)
    # if target should be added to all target
    if(NOT ${_ARG_NO_ALL})
      # set variable for adding target to all target
      set(_all ALL)
    endif()
    # add overall custom cmake target to build videantis target
    add_custom_target(build_${_target} ${_all}
      DEPENDS ${_working_dir}/${_map_file} ${_working_dir}/${_bin_file} ${_working_dir}/${_bin2h_file})
  endif()

  # set property for map and bin file of this target
  set_property(GLOBAL PROPERTY VID_TARGET_${_target_upper}_MAP_FILE ${VID_TARGET_${_target_upper}_WORKING_DIR}/${_map_file})
  set_property(GLOBAL PROPERTY VID_TARGET_${_target_upper}_BIN_FILE ${VID_TARGET_${_target_upper}_WORKING_DIR}/${_bin_file})

  # adapt path of vspasm outputs, binary outputs and bin2h outputs (vspgcc outputs are already adapted),
  # if working directory is not current binary directory
  if(NOT ${_working_dir} STREQUAL ".")
    list(TRANSFORM VID_TARGET_${_target_upper}_VSPASM_OUTPUTS PREPEND ${_working_dir}/)
    list(TRANSFORM VID_TARGET_${_target_upper}_BINARY_OUTPUTS PREPEND ${_working_dir}/)
    list(TRANSFORM VID_TARGET_${_target_upper}_BIN2H_OUTPUTS PREPEND ${_working_dir}/)
  endif()

  # set combined list with all external sources (outputs form commands that are sources to other commands are not included)
  set(VID_TARGET_${_target_upper}_SOURCES ${VID_TARGET_${_target_upper}_VSPGCC_SOURCES} ${VID_TARGET_${_target_upper}_VSPASM_SOURCES})
  # set combined list with all generated outputs
  set(VID_TARGET_${_target_upper}_OUTPUTS ${VID_TARGET_${_target_upper}_VSPGCC_OUTPUTS} ${VID_TARGET_${_target_upper}_VSPASM_OUTPUTS}
                                          ${VID_TARGET_${_target_upper}_BINARY_OUTPUTS} ${VID_TARGET_${_target_upper}_BIN2H_OUTPUTS})
  # set vspgcc commands to commands list
  set(VID_TARGET_${_target_upper}_COMMANDS ${VID_TARGET_${_target_upper}_VSPGCC_COMMANDS})
  # get length of commands list
  list(LENGTH VID_TARGET_${_target_upper}_COMMANDS _cmd_length)
  # if list of commands contains something add && to list
  if(${_cmd_length} GREATER "0")
    list(APPEND VID_TARGET_${_target_upper}_COMMANDS &&)
  endif()
  # append vspasm and binary commands to commands list
  list(APPEND VID_TARGET_${_target_upper}_COMMANDS ${VID_TARGET_${_target_upper}_VSPASM_COMMANDS} && ${VID_TARGET_${_target_upper}_BINARY_COMMANDS})
  # check if a header file with binary code should be created and append command to commands list
  if(${_ARG_BIN2H})
    list(APPEND VID_TARGET_${_target_upper}_COMMANDS && ${VID_TARGET_${_target_upper}_BIN2H_COMMANDS})
  endif()

  # add dependencies to link this target to others
  if(DEFINED _ARG_DEPENDENCIES AND (NOT ${_ARG_NO_TARGET}))
    add_dependencies(build_${_target} ${_ARG_DEPENDENCIES})
  endif()
endmacro(vid_build_target_vsp)
