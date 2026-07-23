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
# DESCRIPTION: Support file for videantis SDK to build code
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

#[=======================================================================[.rst:
.. cmake:command:: vid_init_build_target

  Macro for initializing a videantis build target.
  Target gets defined and some default values get initialized.

  .. code-block:: cmake

    vid_init_build_target(name [VMP_BOOT_ADDRESS vmp_boot_address] [VMP_LLVM_STACK_SIZE vmp_llvm_stack_size])

  Function parameters:

  ``name``
    name of the target to initialize
  ``VMP_BOOT_ADDRESS``
    v-MP boot address, defaults to 0x60000000
  ``VMP_LLVM_STACK_SIZE``
    size of llvm stack

  Exposed global properties:

  ``VID_TARGET_<TARGET_NAME>``
    name of build target (case sensitive)
  ``VID_TARGET_<TARGET_NAME>_TYPE``
    type of build target (VMP)
  ``VID_TARGET_<TARGET_NAME>_TARGET_CPU``
    cpu of the target (e.g. mp4.0)
  ``VID_TARGET_<TARGET_NAME>_TARGET_SOC``
    SoC of the target

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>``
    name of build target (variable only used to ensure other macro are called in the correct scope)
  ``VID_TARGET_<TARGET_NAME>_MP_VERSION``
    v-MP version depending on defined target cpu
  ``VID_TARGET_<TARGET_NAME>_SIZE_MEMORY_DMEM``
    size of DMEM in bytes (default: 4096)
  ``VID_TARGET_<TARGET_NAME>_SIZE_MEMORY_DMEM2``
    size of DMEM2 in bytes (default: 4096 (< mp4.0) | 24576 (>= mp4.0))
  ``VID_TARGET_<TARGET_NAME>_SIZE_MEMORY_DMEM3``
    size of DMEM3 in bytes (default: 32768; if available v-MP 4.0 or newer)
  ``VID_TARGET_<TARGET_NAME>_LLLIB_VMP_LIB_DIR``
    lowlevel library dir for assembler files
  ``VID_TARGET_<TARGET_NAME>_VMPCC_INCLUDES``
    list of include dirs for vmpcc
  ``VID_TARGET_<TARGET_NAME>_VMPASM_INCLUDES``
    list of include dirs for vmpasm
  ``VID_TARGET_<TARGET_NAME>_VMP_LIB_FILES``
    assembler files of lllib (default: vid_vmp_boot_loader.asm, vid_vmp_lib.asm and vid_vmp_mbox.asm)
  ``VID_TARGET_<TARGET_NAME>_VMPCC_OPTIMIZATION_LEVEL``
    vmpcc optimization level (default: -O3)
  ``VID_TARGET_<TARGET_NAME>_VMPCC_TARGET_FLAGS``
    list with cpu and soc flags for vmpcc
  ``VID_TARGET_<TARGET_NAME>_VMPASM_TARGET_FLAGS``
    list with cpu and soc flags for vmpasm
  ``VID_TARGET_<TARGET_NAME>_VMPASM_STATISTIC_FLAGS``
    vmpasm statistic flag (default: --statistic=MEMORYIMAGE,OVERLAY)
  ``VID_TARGET_<TARGET_NAME>_VMP_LLVM_STACK_SIZE``
    llvm stack size in 64-bit words (if VMP_LLVM_STACK_SIZE is set when calling the marco)
  ``VID_TARGET_<TARGET_NAME>_VMPASM_OLM_BOOTADDR_FLAGS``
    vmpasm olm boot address flag (default: --olm-initial-in-hardware-memory=<sdram_addr>)
  ``VID_TARGET_<TARGET_NAME>_VMPASM_HW_SOURCECODE_INFO_FLAGS``
    vmpasm hw source code info flags (default: --enable-hw-sourcecode-info,
    --enable-absolute-sourcecode-info and
    --enable-comment-output)
  ``VID_TARGET_<TARGET_NAME>_VMPASM_OUT_FLAGS``
    vmpasm out flags (dmem3 flag if v-MP 4.0 or newer)
    (default: --out-asm=<target_name>.out.asm,
    --out-ext-asm=<target_name>.ovl.out.asm,
    --out-code=<target_name>.imem,
    --out-ext-code=<target_name>.ovl.imem,
    --out-nice-memory-imem=<target_name>.niceout.asm,
    --out-hw-asm=<target_name>.hw.asm,
    --out-memory-map=<target_name>.nocodemap,
    --out-memory-dmem=<target_name>.dmem,
    --out-memory-dmem2=<target_name>.dmem2 and
    --out-memory-dmem3=<target_name>.dmem3)
  ``VID_TARGET_<TARGET_NAME>_VMPASM_CUSTOM_FLAGS``
    variable to add custom flags for vmpasm
  ``VID_TARGET_<TARGET_NAME>_VMPASM_DONT_ELIMINATE_DMA_LEGACY``
    variable to set --dont-eliminate-sections vmpasm option for dma_legacy dsection (if v-MP 4.0 or newer)
  ``VID_TARGET_<TARGET_NAME>_VMPASM_DONT_ELIMINATE_EDMA_DESCR``
    variable to set --dont-eliminate-sections vmpasm option for edma_descr dsection (if v-MP 4.0 or newer)
  ``VID_TARGET_<TARGET_NAME>_VMP_LLVM_STACK_MEM``
    variable to overwrite placement memory of v-MP llvm stack
  ``VID_TARGET_<TARGET_NAME>_VMP_LLVM_STACK_ORG``
    variable to overwrite placement address of v-MP llvm stack
  ``VID_TARGET_<TARGET_NAME>_VMP_MBOX_MEM``
    variable to overwrite placement memory of v-MP mbox segment
  ``VID_TARGET_<TARGET_NAME>_VMP_OLM_MEM``
    variable to overwrite placement memory of v-MP overlay manager segment
  ``VID_TARGET_<TARGET_NAME>_VMP_NUM_DMA_CHANNELS``
    variable to overwrite number of DMA channels
  ``VID_TARGET_<TARGET_NAME>_VMP_DMA_DESCR_MEM``
    variable to overwrite placement memory of v-MP DMA descriptors
  ``VID_TARGET_<TARGET_NAME>_VMP_DMA_DESCR_ORG``
    variable to overwrite placement address of v-MP DMA descriptors
  ``VID_TARGET_<TARGET_NAME>_VMP_DMA_LEGACY_ORG``
    variable to overwrite placement address of v-MP DMA legacy high address part segment
  ``VID_TARGET_<TARGET_NAME>_VMP_DMA_LEGACY_HIGH``
    variable to overwrite v-MP DMA legacy high address part
  ``VID_TARGET_<TARGET_NAME>_VMP_EDMA_DESCR_MEM``
    variable to overwrite placement memory of v-MP EDMA descriptors
  ``VID_TARGET_<TARGET_NAME>_VMP_EDMA_DESCR_ORG``
    variable to overwrite placement address of v-MP EDMA descriptors

  .. note::

    Some of the variables have a dependency on each other.
    E.g. changing VID_TARGET_<TARGET_NAME>_VMP_LLVM_STACK_MEM requires a change of
    VID_TARGET_<TARGET_NAME>_VMP_LLVM_STACK_ORG, because the default address won't
    work anymore.

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<target_name>``
    target name
  ``<sdram_addr>```
    SDRAM base address of the lowlevel library (defined as target constant)
#]=======================================================================]
macro(vid_init_build_target _name)
  cmake_parse_arguments(_ARG "" "VMP_BOOT_ADDRESS;VMP_LLVM_STACK_SIZE"
    "" ${ARGN})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_init_build_target() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
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
  set_property(GLOBAL PROPERTY VID_TARGET_${_name_upper}_TYPE VMP)

  # get global defined lllib
  unset(_lllib)
  get_property(_lllib GLOBAL PROPERTY VID_TARGET_LLLIB)

  # check if lllib was set
  if(NOT DEFINED _lllib)
    message(FATAL_ERROR "No videantis lowlevel library is specified! "
      "Define a project wide lowlevel library with vid_define_system_vmp().")
  endif()

  # get global defined target cpu
  unset(_target_cpu)
  get_property(_target_cpu GLOBAL PROPERTY VID_TARGET_TARGET_CPU_MP)

  # check if target cpu was set
  if(NOT DEFINED _target_cpu)
    message(FATAL_ERROR "No target cpu is specified! "
      "Define a project wide target cpu with vid_define_system_vmp().")
  endif()

  # set global property with target cpu for this build target
  set_property(GLOBAL PROPERTY VID_TARGET_${_name_upper}_TARGET_CPU ${_target_cpu})

  # get global defined target soc
  unset(_target_soc)
  get_property(_target_soc GLOBAL PROPERTY VID_TARGET_TARGET_SOC_MP)

  # check if target soc was set
  if(NOT DEFINED _target_soc)
    message(FATAL_ERROR "No target soc is specified! "
      "Define a project wide target soc with vid_define_system_vmp().")
  endif()

  # set global property with target soc for this build target
  set_property(GLOBAL PROPERTY VID_TARGET_${_name_upper}_TARGET_SOC ${_target_soc})

  # check if requested lowlevel library is available in the used SDK
  if(NOT ${_lllib} IN_LIST VIDSDK_LOWLEVELLIBRARY)
    message(FATAL_ERROR "${_lllib} is not available in ${VIDSDK}")
  endif()

  # get v-MP version based on target_cpu
  string(REPLACE "mp" "" VID_TARGET_${_name_upper}_MP_VERSION ${_target_cpu})

  # set default sizes for data memories based on v-MP version
  set(VID_TARGET_${_name_upper}_SIZE_MEMORY_DMEM 4096)
  set(VID_TARGET_${_name_upper}_SIZE_MEMORY_DMEM2 4096)
  if(${VID_TARGET_${_name_upper}_MP_VERSION} VERSION_GREATER_EQUAL "4.0")
    set(VID_TARGET_${_name_upper}_SIZE_MEMORY_DMEM2 24576)
    set(VID_TARGET_${_name_upper}_SIZE_MEMORY_DMEM3 32768)
  endif()

  # get lowlevel library in upper case letters
  string(TOUPPER ${_lllib} _lllib_upper)

  # check if lowlevel library include dirs are defined
  if(NOT DEFINED VID_${_lllib_upper}_HOST_INCLUDE_DIRS)
    message(FATAL_ERROR "Cmake variable VID_${_lllib_upper}_HOST_INCLUDE_DIRS is not defined.")
  endif()

  # check if lowlevel library v-MP library dir is defined
  if(NOT DEFINED VID_${_lllib_upper}_VMP_LIB_DIR)
    message(FATAL_ERROR "Cmake variable VID_${_lllib_upper}_VMP_LIB_DIR is not defined.")
  endif()

  # initialize vmpcc includes with v-MP include dir of lllib
  set(VID_TARGET_${_name_upper}_VMPCC_INCLUDES ${VID_${_lllib_upper}_HOST_INCLUDE_DIRS})

  # get lowlevel library v-MP lib dir
  set(VID_TARGET_${_name_upper}_LLLIB_VMP_LIB_DIR ${VID_${_lllib_upper}_VMP_LIB_DIR})

  # initialize vmpasm includes with v-MP lib dir of lllib and current binary dir
  # (compiled *.cl sources will be found in the current binary dir)
  set(VID_TARGET_${_name_upper}_VMPASM_INCLUDES ${VID_TARGET_${_name_upper}_LLLIB_VMP_LIB_DIR} ${CMAKE_CURRENT_BINARY_DIR})

  # set default library files for v-MP coming from the SDK
  set(VID_TARGET_${_name_upper}_VMP_LIB_FILES vid_vmp_boot_loader.asm vid_vmp_lib.asm vid_vmp_mbox.asm)

  # set default vmpcc optimization level
  set(VID_TARGET_${_name_upper}_VMPCC_OPTIMIZATION_LEVEL "-O3")

  # create target compile flags based on target_cpu and target_soc
  set(VID_TARGET_${_name_upper}_VMPCC_TARGET_FLAGS "-mcpu=${_target_cpu}" "-vmp-soc=${_target_soc}")

  # create target assemble flags based on target_cpu and target_soc
  set(VID_TARGET_${_name_upper}_VMPASM_TARGET_FLAGS "--cpu=${_target_cpu}" "--soc=${_target_soc}")

  # set default vmpasm statistic flags
  set(VID_TARGET_${_name_upper}_VMPASM_STATISTIC_FLAGS "--statistic=MEMORYIMAGE,OVERLAY")

  # define VMP_LLVM_STACK_SIZE when argument is given
  if(DEFINED _ARG_VMP_LLVM_STACK_SIZE)
    set(VID_TARGET_${_name_upper}_VMP_LLVM_STACK_SIZE ${_ARG_VMP_LLVM_STACK_SIZE})
  endif()

  # get SDRAM address from global property and set default v-MP boot address
  unset(_vmp_boot_address)
  get_property(_vmp_boot_address GLOBAL PROPERTY VID_TARGET_${_lllib_upper}_SDRAM_ADDR)

  # check if SDRAM address was set
  if(NOT DEFINED _vmp_boot_address)
    message(FATAL_ERROR "No SDRAM address is specified! "
      "Get a project wide SDRAM address with vid_define_system_vmp().")
  endif()

  # if v-MP boot address is defined when calling this macro overwrite default value
  if(DEFINED _ARG_VMP_BOOT_ADDRESS)
    set(_vmp_boot_address ${_ARG_VMP_BOOT_ADDRESS})
  endif()
  # set default vmpasm olm boot address flags
  set(VID_TARGET_${_name_upper}_VMPASM_OLM_BOOTADDR_FLAGS "--olm-initial-in-hardware-memory=${_vmp_boot_address}")

  # set default vmpasm hw source code info flags
  set(VID_TARGET_${_name_upper}_VMPASM_HW_SOURCECODE_INFO_FLAGS "--enable-hw-sourcecode-info"
                                                                "--enable-absolute-sourcecode-info"
                                                                "--enable-comment-output")

  # set default vmpasm out flags
  set(VID_TARGET_${_name_upper}_VMPASM_OUT_FLAGS "--out-asm=${_name}.out.asm"
                                                 "--out-ext-asm=${_name}.ovl.out.asm"
                                                 "--out-code=${_name}.imem"
                                                 "--out-ext-code=${_name}.ovl.imem"
                                                 "--out-nice-memory-imem=${_name}.niceout.asm"
                                                 "--out-hw-asm=${_name}.hw.asm"
                                                 "--out-memory-map=${_name}.nocodemap"
                                                 "--out-memory-dmem=${_name}.dmem"
                                                 "--out-memory-dmem2=${_name}.dmem2")
  # if v-MP 4.0 or newer attach dmem3 out memory flag
  # and create variable to preserve the dsections dma_legacy and edma_descr
  if(${VID_TARGET_${_name_upper}_MP_VERSION} VERSION_GREATER_EQUAL "4.0")
    list(APPEND VID_TARGET_${_name_upper}_VMPASM_OUT_FLAGS "--out-memory-dmem3=${_name}.dmem3")

    # create variable to preserve the dsection dma_legacy
    set(VID_TARGET_${_name_upper}_VMPASM_DONT_ELIMINATE_DMA_LEGACY --dont-eliminate-sections=dma_legacy)
    # create variable to preserve the dsection edma_descr
    set(VID_TARGET_${_name_upper}_VMPASM_DONT_ELIMINATE_EDMA_DESCR --dont-eliminate-sections=edma_descr)
  endif()
endmacro(vid_init_build_target)

#[=======================================================================[.rst:
.. cmake:command:: vid_init_build_target_vmp

  Macro for initializing a videantis v-MP build target.
  This is a wrapper macro for :cmake:command:`vid_init_build_target`.

  .. code-block:: cmake

    vid_init_build_target_vmp(name [VMP_BOOT_ADDRESS vmp_boot_address] [VMP_LLVM_STACK_SIZE vmp_llvm_stack_size])

  Function parameters:

  ``name``
    name of the target to initialize
  ``VMP_BOOT_ADDRESS``
    v-MP boot address, defaults to 0x60000000
  ``VMP_LLVM_STACK_SIZE``
    size of llvm stack
#]=======================================================================]
macro(vid_init_build_target_vmp _name)
  cmake_parse_arguments(_ARG "" "VMP_BOOT_ADDRESS;VMP_LLVM_STACK_SIZE"
    "" ${ARGN})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_init_build_target_vmp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  vid_init_build_target(${_name} VMP_BOOT_ADDRESS ${_ARG_VMP_BOOT_ADDRESS} VMP_LLVM_STACK_SIZE ${_ARG_VMP_LLVM_STACK_SIZE})
endmacro(vid_init_build_target_vmp)

#[=======================================================================[.rst:
.. cmake:command:: _add_sources

  Internal macro for adding source files to a videantis build target.
  Don't call this macro directly!

  .. code-block:: cmake

    _add_sources(target compiler assembler [IDENTIFIERS id1 [id2 ...] [FILES file1 [file2 ...]]]
        [ADD file1 [file2 ...]] [PRINT_AVAILABLE_SOURCES])

  Function parameters:

  ``target``
    name of an already defined build target
  ``compiler``
    name of the compiler used in tool chain
  ``assembler``
    name of the assembler used in tool chain
  ``IDENTIFIERS``
    list of identifiers for source files
  ``FILES``
    list of files from identifier

  .. note::

    Only one identifier is allow when FILES is set (:cmake:command:`vid_add_sources` can be called multiple times)

  ``ADD``
    list of files add as sources without registering as globally available file
  ``PRINT_AVAILABLE_SOURCES``
    print globally available sources

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_<COMPILER>_SOURCES``
    list of source files for the compiler with full path
  ``VID_TARGET_<TARGET_NAME>_<COMPILER>_OUTPUTS``
    list of generated files of the compiler (<target_name>_[<identifier>_]<source_filename_we>.asm)

  .. note::

    lists of compiler sources and outputs are mapped lists (same index for belonging files)

  ``VID_TARGET_<TARGET_NAME>_<ASSEMBLER>_SOURCES``
    list of source files for the assembler with full path
  ``VID_TARGET_<TARGET_NAME>_VIDASM_HELPER_FILES``
    list of files that can generate an additional vidasm helper file as vmpcc output
  ``VID_TARGET_<TARGET_NAME>_<SOURCE_FILENAME_WE>_VIDASM_HELPER_FILE``
    variable to enable generation of vidasm helper file for a vmpcc source (default: FALSE)
  ``VID_TARGET_<TARGET_NAME>_<IDENTIFIER>_<SOURCE_FILENAME_WE>_VIDASM_HELPER_FILE``
    variable to enable generation of vidasm helper file for a vmpcc source from identifier (default: FALSE)

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<target_name>``
    target name
  ``<COMPILER>``
    name of the compiler in all upper case letters
  ``<ASSEMBLER>``
    name of the assembler in all upper case letters
  ``<IDENTIFIER>``
    identifier in all upper case letters if file was globally added with :cmake:command:`vid_register_file`
  ``<identifier>``
    identifier if file was globally added with :cmake:command:`vid_register_file`
  ``<SOURCE_FILENAME_WE>``
    source file name without file extension in all upper case letters
  ``<source_filename_we>``
    source file name without file extension
#]=======================================================================]
macro(_add_sources _target _compiler _assembler)
  cmake_parse_arguments(_ARG "PRINT_AVAILABLE_SOURCES" ""
    "IDENTIFIERS;FILES;ADD" ${ARGN})

  # get all upper case strings for target, compiler and assembler
  string(TOUPPER ${_target} _target_upper)
  string(TOUPPER ${_compiler} _compiler_upper)
  string(TOUPPER ${_assembler} _assembler_upper)

  # get lists of registered files compiler/assembler
  get_property(_vid_registered_files_compiler GLOBAL PROPERTY VID_REGISTERED_FILES_${_compiler_upper})
  get_property(_vid_registered_files_assembler GLOBAL PROPERTY VID_REGISTERED_FILES_${_assembler_upper})

  # if PRINT_AVAILABLE_SOURCES is set, output videantis build process registered sources
  if(${_ARG_PRINT_AVAILABLE_SOURCES})
    message(STATUS "VID_REGISTERED_FILES_${_compiler_upper}: ${_vid_registered_files_compiler}")

    foreach(_file_identifier ${_vid_registered_files_compiler})
      get_property(_file_path GLOBAL PROPERTY ${_file_identifier}_${_compiler_upper}_PATH)
      get_property(_file_id GLOBAL PROPERTY ${_file_identifier}_${_compiler_upper}_ID)
      message(STATUS "${_file_identifier}_${_compiler_upper}_PATH: ${_file_path} (${_file_id})")
    endforeach(_file_identifier)

    message(STATUS "VID_REGISTERED_FILES_${_assembler_upper}: ${_vid_registered_files_assembler}")

    foreach(_file_identifier ${_vid_registered_files_assembler})
      get_property(_file_path GLOBAL PROPERTY ${_file_identifier}_${_assembler_upper}_PATH)
      get_property(_file_id GLOBAL PROPERTY ${_file_identifier}_${_assembler_upper}_ID)
      message(STATUS "${_file_identifier}_${_assembler_upper}_PATH: ${_file_path} (${_file_id})")
    endforeach(_file_identifier)
  endif()

  # clear internal variable from previous calls of _add_sources()
  unset(_list_files)

  # iterate over files to add
  foreach(_file ${_ARG_ADD})
    # check if a file with absolute path is given
    # otherwise set absolute path
    if(IS_ABSOLUTE ${_file})
      set(_file_path ${_file})
    else()
      get_filename_component(_file_path ${_file} ABSOLUTE)
    endif()

    # check if specified file exists
    if(NOT EXISTS ${_file_path})
      message(FATAL_ERROR "File ${_file_path} does not exist")
    endif()

    # extract file extension in upper case letters
    string(REGEX MATCH "\\.([a-zA-Z]*)$" _ ${_file})
    string(TOUPPER ${CMAKE_MATCH_1} _file_ext)

    # check for correct file extension and set related tool
    if(${_file_ext} IN_LIST ${_compiler_upper}_SUPPORTED_FILE_TYPES)
      set(_tool ${_compiler_upper})
    elseif(${_file_ext} IN_LIST ${_assembler_upper}_SUPPORTED_FILE_TYPES)
      set(_tool ${_assembler_upper})
    else()
      message(FATAL_ERROR "Only ${${_compiler_upper}_SUPPORTED_FILE_TYPES} and ${${_assembler_upper}_SUPPORTED_FILE_TYPES} files are supported")
    endif()

    # append the file to list of sources
    list(APPEND VID_TARGET_${_target_upper}_${_tool}_SOURCES ${_file_path})
    # if tool is a compiler create a unique mapped output filename
    if(${_tool} STREQUAL ${_compiler_upper})
      # extract filename
      get_filename_component(_filename_we ${_file} NAME_WE)
      # append a generated filename to a list of mapped outputs
      list(APPEND VID_TARGET_${_target_upper}_${_tool}_OUTPUTS ${_target}_${_filename_we}.asm)
      # if _add_sources is called for v-MP (vmpcc)
      if(${_compiler} STREQUAL "vmpcc")
        # get the filename in all upper case letters
        string(TOUPPER ${_filename_we} _filename_we_upper)
        # set variable for creating vidasm helper file for this vmpcc output to default false
        set(VID_TARGET_${_target_upper}_${_filename_we_upper}_VIDASM_HELPER_FILE FALSE)
        # add variable name to to list of vidasm helper files
        list(APPEND VID_TARGET_${_target_upper}_VIDASM_HELPER_FILES VID_TARGET_${_target_upper}_${_filename_we_upper}_VIDASM_HELPER_FILE)
      endif()
    endif()
  endforeach(_file)

  # iterate over files from identifier
  foreach(_file ${_ARG_FILES})
    # get file name
    get_filename_component(_filename ${_file} NAME)

    # test if no path is specified
    if(NOT ${_filename} STREQUAL ${_file})
      message(WARNING "Argument FILES only checks for filename for the given IDENTIFIER. "
        "Appended paths to the filename will be ignored.")
    endif()

    # extract file extension in upper case letters
    string(REGEX MATCH "\\.([a-zA-Z]*)$" _ ${_file})
    string(TOUPPER ${CMAKE_MATCH_1} _file_ext)

    # unset variable to avoid previously set variable from other macro call
    unset(_vid_registered_files)
    # set registered files list based on file extension
    if(${_file_ext} IN_LIST ${_compiler_upper}_SUPPORTED_FILE_TYPES)
      set(_vid_registered_files ${_vid_registered_files_compiler})
    elseif(${_file_ext} IN_LIST ${_assembler_upper}_SUPPORTED_FILE_TYPES)
      set(_vid_registered_files ${_vid_registered_files_assembler})
    endif()

    # extract filename without extension
    get_filename_component(_filename_we ${_file} NAME_WE)

    # create file identifier
    string(TOUPPER ${_ARG_IDENTIFIERS}_${_filename_we} _file_identifier)

    # check if file identifier is registered
    if(NOT ${_file_identifier} IN_LIST _vid_registered_files)
      message(FATAL_ERROR "${_file} is not registered for identifier ${_ARG_IDENTIFIERS}")
    endif()

    # append filename to a list of files
    list(APPEND _list_files ${_filename})
  endforeach(_file)

  # iterate over identifiers
  foreach(_identifier ${_ARG_IDENTIFIERS})
    # set current identifier with upper case
    string(TOUPPER ${_identifier} _identifier_upper)

    # use reverse index to get only files registered under this identifier (O(k) vs O(N) scan)
    get_property(_identifier_files_compiler GLOBAL PROPERTY VID_REGISTERED_FILES_${_compiler_upper}_ID_${_identifier_upper})
    foreach(_file ${_identifier_files_compiler})
      # get full path of registered compiler file
      get_property(_file_path GLOBAL PROPERTY ${_file}_${_compiler_upper}_PATH)
      # extract filename without extension
      get_filename_component(_filename_we ${_file_path} NAME_WE)

      # check if only specific files should be added
      if(DEFINED _ARG_FILES)
        # get filename of registered file from full path
        get_filename_component(_filename ${_file_path} NAME)

        # check if registered filename is in list of needed files
        if(${_filename} IN_LIST _list_files)
          # append the file to list of sources
          list(APPEND VID_TARGET_${_target_upper}_${_compiler_upper}_SOURCES ${_file_path})
          # append a generated filename to a list of mapped outputs
          list(APPEND VID_TARGET_${_target_upper}_${_compiler_upper}_OUTPUTS ${_target}_${_identifier}_${_filename_we}.asm)
          # if _add_sources is called for v-MP (vmpcc)
          if(${_compiler} STREQUAL "vmpcc")
            # get the filename in all upper case letters
            string(TOUPPER ${_filename_we} _filename_we_upper)
            # set variable for creating vidasm helper file for this vmpcc output to default false
            set(VID_TARGET_${_target_upper}_${_identifier_upper}_${_filename_we_upper}_VIDASM_HELPER_FILE FALSE)
            # add variable name to to list of vidasm helper files
            list(APPEND VID_TARGET_${_target_upper}_VIDASM_HELPER_FILES VID_TARGET_${_target_upper}_${_identifier_upper}_${_filename_we_upper}_VIDASM_HELPER_FILE)
          endif()
        endif()
      # all files of an identifier should be added
      else()
        # append the file to list of sources
        list(APPEND VID_TARGET_${_target_upper}_${_compiler_upper}_SOURCES ${_file_path})
        # append a generated filename to a list of mapped outputs
        list(APPEND VID_TARGET_${_target_upper}_${_compiler_upper}_OUTPUTS ${_target}_${_identifier}_${_filename_we}.asm)
        # if _add_sources is called for v-MP (vmpcc)
        if(${_compiler} STREQUAL "vmpcc")
          # get the filename in all upper case letters
          string(TOUPPER ${_filename_we} _filename_we_upper)
          # set variable for creating vidasm helper file for this vmpcc output to default false
          set(VID_TARGET_${_target_upper}_${_identifier_upper}_${_filename_we_upper}_VIDASM_HELPER_FILE FALSE)
          # add variable name to to list of vidasm helper files
          list(APPEND VID_TARGET_${_target_upper}_VIDASM_HELPER_FILES VID_TARGET_${_target_upper}_${_identifier_upper}_${_filename_we_upper}_VIDASM_HELPER_FILE)
        endif()
      endif()
    endforeach(_file)

    # use reverse index to get only assembler files registered under this identifier (O(k) vs O(N) scan)
    get_property(_identifier_files_assembler GLOBAL PROPERTY VID_REGISTERED_FILES_${_assembler_upper}_ID_${_identifier_upper})
    foreach(_file ${_identifier_files_assembler})
      # get full path of registered assembler file
      get_property(_file_path GLOBAL PROPERTY ${_file}_${_assembler_upper}_PATH)

      # check if only specific files should be added
      if(DEFINED _ARG_FILES)
        # get filename of registered file from full path
        get_filename_component(_filename ${_file_path} NAME)

        # check if registered filename is in list of needed files
        if(${_filename} IN_LIST _list_files)
          # append the file to list of sources
          list(APPEND VID_TARGET_${_target_upper}_${_assembler_upper}_SOURCES ${_file_path})
        endif()
      # all files of an identifier should be added
      else()
        # append the file to list of sources
        list(APPEND VID_TARGET_${_target_upper}_${_assembler_upper}_SOURCES ${_file_path})
      endif()
    endforeach(_file)
  endforeach(_identifier)
endmacro(_add_sources)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_sources

  Macro for adding source files to a videantis build target.
  This macro requires at least one IDENTIFIER to add source files
  or one source file added as local source file via ADD.
  If not all source files of one IDENTIFIER should be added,
  the files can be selected with FILES.

  .. code-block:: cmake

    vid_add_sources(target [IDENTIFIERS id1 [id2 ...] [FILES file1 [file2 ...]]]
        [ADD file1 [file2 ...]] [PRINT_AVAILABLE_SOURCES])

  Function parameters:

  ``target``
    name of an already defined build target
  ``IDENTIFIERS``
    list of identifiers for source files
  ``FILES``
    list of files from identifier

  .. note::

    Only one identifier is allow when FILES is set (:cmake:command:`vid_add_sources` can be called multiple times)

  ``ADD``
    list of files add as sources without registering as globally available file
  ``PRINT_AVAILABLE_SOURCES``
    print globally available sources

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VMPCC_SOURCES``
    list of source files for vmpcc with full path
  ``VID_TARGET_<TARGET_NAME>_VMPCC_OUTPUTS``
    list of generated files of vmpcc (<target_name>_[<identifier>_]<source_filename_we>.asm)

  .. note::

    lists of vmpcc sources and outputs are mapped lists (same index for belonging files)

  ``VID_TARGET_<TARGET_NAME>_VMPASM_SOURCES``
    list of source files for vmpasm with full path
  ``VID_TARGET_<TARGET_NAME>_VIDASM_HELPER_FILES``
    list of files that can generate an additional vidasm helper file as vmpcc output
  ``VID_TARGET_<TARGET_NAME>_<SOURCE_FILENAME_WE>_VIDASM_HELPER_FILE``
    variable to enable generation of vidasm helper file for a vmpcc source (default: FALSE)
  ``VID_TARGET_<TARGET_NAME>_<IDENTIFIER>_<SOURCE_FILENAME_WE>_VIDASM_HELPER_FILE``
    variable to enable generation of vidasm helper file for a vmpcc source from identifier (default: FALSE)

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<target_name>``
    target name
  ``<IDENTIFIER>``
    identifier in all upper case letters if file was globally added with :cmake:command:`vid_register_file`
  ``<identifier>``
    identifier if file was globally added with :cmake:command:`vid_register_file`
  ``<SOURCE_FILENAME_WE>``
    source file name without file extension in all upper case letters
  ``<source_filename_we>``
    source file name without file extension
#]=======================================================================]
macro(vid_add_sources _target)
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
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_add_sources() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VMP")
    message(FATAL_ERROR "vid_add_sources() needs to be called for a v-MP target")
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
    message(FATAL_ERROR "vid_add_sources() requires minimum one IDENTIFIER or one file via ADD")
  endif()
  # when sources are selected with FILES only one IDENTIFIER is allowed
  if((${_num_identifiers} GREATER 1) AND (DEFINED _ARG_FILES))
    message(FATAL_ERROR "vid_add_sources() can take only multiple IDENTIFIERs, when no FILES sources are specified")
  endif()
  if((DEFINED _ARG_FILES) AND (${_num_identifiers} EQUAL 0))
    message(FATAL_ERROR "vid_add_sources() needs one IDENTIFIER, when FILES sources are specified")
  endif()

  # unset print_available_sources from possible prior calls of vid_add_sources()
  unset(_print_available_sources)
  # check if PRINT_AVAILABLE_SOURCES is set
  if(${_ARG_PRINT_AVAILABLE_SOURCES})
    set(_print_available_sources PRINT_AVAILABLE_SOURCES)
  endif()

  # call internal macro to add sources
  _add_sources(${_target} vmpcc vmpasm IDENTIFIERS ${_ARG_IDENTIFIERS} FILES ${_ARG_FILES} ADD ${_ARG_ADD} ${_print_available_sources})
endmacro(vid_add_sources)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_sources_vmp

  Macro for adding source files to a videantis v-MP build target.
  This is a wrapper macro for :cmake:command:`vid_add_sources`.

  .. code-block:: cmake

    vid_add_sources_vmp(target [IDENTIFIERS id1 [id2 ...] [FILES file1 [file2 ...]]]
        [ADD file1 [file2 ...]] [PRINT_AVAILABLE_SOURCES])

  Function parameters:

  ``target``
    name of an already defined build target
  ``IDENTIFIERS``
    list of identifiers for source files
  ``FILES``
    list of files from identifier

  .. note::

    Only one identifier is allow when FILES is set (:cmake:command:`vid_add_sources_vmp` can be called multiple times)


  ``ADD``
    list of files add as sources without registering as globally available file
  ``PRINT_AVAILABLE_SOURCES``
    print globally available sources
#]=======================================================================]
macro(vid_add_sources_vmp _target)
  cmake_parse_arguments(_ARG "PRINT_AVAILABLE_SOURCES" ""
    "IDENTIFIERS;FILES;ADD" ${ARGN})

  # unset print_available_sources from possible prior calls of vid_add_sources_vmp()
  unset(_print_available_sources)
  # check if PRINT_AVAILABLE_SOURCES is set
  if(${_ARG_PRINT_AVAILABLE_SOURCES})
    set(_print_available_sources PRINT_AVAILABLE_SOURCES)
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_sources_vmp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  vid_add_sources(${_target} IDENTIFIERS ${_ARG_IDENTIFIERS} FILES ${_ARG_FILES} ADD ${_ARG_ADD} ${_print_available_sources})
endmacro(vid_add_sources_vmp)

#[=======================================================================[.rst:
.. cmake:command:: _add_includes

  Internal macro for adding include directories to a videantis build target.
  Don't call this macro directly!

  .. code-block:: cmake

    _add_includes(target compiler assembler [IDENTIFIERS id1 [id2 ...]]
        [DIRECTORIES dir1 [dir2 ...]] [DIRECTORIES_<ASSEMBLER> dir1 [dir2 ...]]
        [ADD dir1 [dir2 ...]] [ADD_<ASSEMBLER> dir1 [dir2 ...]] [PRINT_AVAILABLE_INCLUDES])

  Function parameters:

  ``target``
    name of an already defined build target
  ``compiler``
    name of the compiler used in tool chain
  ``assembler``
    name of the assembler used in tool chain
  ``IDENTIFIERS``
    list of identifiers for include dirs
  ``DIRECTORIES``
    list of dirs from the identifier (no full path)
  ``DIRECTORIES_<ASSEMBLER>``
    list of dirs for the assembler from the identifier (no full path)

  .. note::

    Only one identifier is allow when DIRECTORIES/DIRECTORIES_<ASSEMBLER> is set (:cmake:command:`vid_add_includes` can be called multiple times).
    If an include directory is registered as . (dot), the character . (dot) has to be used to select it.

  ``ADD``
    list of dirs add as includes without registering as globally available dir
  ``ADD_<ASSEMBLER>``
    list of dirs add as includes for the assembler without registering as globally available dir
  ``PRINT_AVAILABLE_INCLUDES``
    print globally available includes

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_<COMPILER>_INCLUDES``
    list of include directories for the compiler with full path
  ``VID_TARGET_<TARGET_NAME>_<ASSEMBLER>_INCLUDES``
    list of include directories for the assembler with full path

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<COMPILER>``
    name of the compiler in all upper case letters
  ``<ASSEMBLER>``
    name of the assembler in all upper case letters
#]=======================================================================]
macro(_add_includes _target _compiler _assembler)
  # get all upper case strings for target, compiler and assembler
  string(TOUPPER ${_target} _target_upper)
  string(TOUPPER ${_compiler} _compiler_upper)
  string(TOUPPER ${_assembler} _assembler_upper)

  cmake_parse_arguments(_ARG "PRINT_AVAILABLE_INCLUDES" ""
    "IDENTIFIERS;DIRECTORIES;DIRECTORIES_${_assembler_upper};ADD;ADD_${_assembler_upper}" ${ARGN})

  # get lists of registered include dirs compiler/assembler
  get_property(_vid_registered_inc_dirs_compiler GLOBAL PROPERTY VID_REGISTERED_INC_DIRS_${_compiler_upper})
  get_property(_vid_registered_inc_dirs_assembler GLOBAL PROPERTY VID_REGISTERED_INC_DIRS_${_assembler_upper})

  # if PRINT_AVAILABLE_INCLUDES is set, output videantis build process registered include dirs
  if(${_ARG_PRINT_AVAILABLE_INCLUDES})
    message(STATUS "VID_REGISTERED_INC_DIRS_${_compiler_upper}: ${_vid_registered_inc_dirs_compiler}")

    foreach(_dir_identifier ${_vid_registered_inc_dirs_compiler})
      get_property(_dir_path GLOBAL PROPERTY ${_dir_identifier}_${_compiler_upper}_PATH)
      get_property(_dir_id GLOBAL PROPERTY ${_dir_identifier}_${_compiler_upper}_ID)
      message(STATUS "${_dir_identifier}_${_compiler_upper}_PATH: ${_dir_path} (${_dir_id})")
    endforeach(_dir_identifier)

    message(STATUS "VID_REGISTERED_INC_DIRS_${_assembler_upper}: ${_vid_registered_inc_dirs_assembler}")

    foreach(_dir_identifier ${_vid_registered_inc_dirs_assembler})
      get_property(_dir_path GLOBAL PROPERTY ${_dir_identifier}_${_assembler_upper}_PATH)
      get_property(_dir_id GLOBAL PROPERTY ${_dir_identifier}_${_assembler_upper}_ID)
      message(STATUS "${_dir_identifier}_${_assembler_upper}_PATH: ${_dir_path} (${_dir_id})")
    endforeach(_dir_identifier)
  endif()

  # clear internal variables from previous calls of _add_includes()
  unset(_list_directories)
  unset(_list_directories_assembler)

  # iterate over files to add
  foreach(_dir ${_ARG_ADD})
    # check if a dir with absolute path is given
    # otherwise set absolute path
    if(IS_ABSOLUTE ${_dir})
      set(_dir_path ${_dir})
    else()
      get_filename_component(_dir_path ${_dir} ABSOLUTE)
    endif()

    # check if specified file exists
    if((NOT EXISTS ${_dir_path}) AND (IS_DIRECTORY ${_dir_path}))
      message(FATAL_ERROR "Directory ${_dir} does not exist")
    endif()

    # append the file to list of sources
    list(APPEND VID_TARGET_${_target_upper}_${_compiler_upper}_INCLUDES ${_dir_path})
  endforeach(_dir)

  # iterate over files to add
  foreach(_dir ${_ARG_ADD_${_assembler_upper}})
    # check if a dir with absolute path is given
    # otherwise set absolute path
    if(IS_ABSOLUTE ${_dir})
      set(_dir_path ${_dir})
    else()
      get_filename_component(_dir_path ${_dir} ABSOLUTE)
    endif()

    # check if specified file exists
    if((NOT EXISTS ${_dir_path}) AND (IS_DIRECTORY ${_dir_path}))
      message(FATAL_ERROR "Directory ${_dir_path} does not exist")
    endif()

    # append the file to list of sources
    list(APPEND VID_TARGET_${_target_upper}_${_assembler_upper}_INCLUDES ${_dir_path})
  endforeach(_dir)

  # only one IDENTIFIER is set (tested before)
  # iterate over directories to add
  foreach(_dir ${_ARG_DIRECTORIES})
    # create dir identifier
    get_filename_component(_dir_name ${_dir} NAME)
    string(TOUPPER ${_ARG_IDENTIFIERS}_${_dir_name} _dir_identifier)
    string(REPLACE "." "DOT" _dir_identifier ${_dir_identifier})

    # test if no path is specified
    if(NOT ${_dir_name} STREQUAL ${_dir})
      message(WARNING "Argument DIRECTORIES only checks for directory name for the given IDENTIFIER. "
        "Appended paths to the directory name will be ignored.")
    endif()

    # check if dir identifier is registered
    if(NOT ${_dir_identifier} IN_LIST _vid_registered_inc_dirs_compiler)
      message(FATAL_ERROR "${_dir} is not registered for identifier ${_ARG_IDENTIFIERS}")
    endif()

    # append created dir identifier to a list of dir identifiers
    list(APPEND _list_directories ${_dir_identifier})
  endforeach(_dir)

  # only one IDENTIFIER is set (tested before)
  # iterate over assembler directories to add
  foreach(_dir ${_ARG_DIRECTORIES_${_assembler_upper}})
    # create dir identifier
    get_filename_component(_dir_name ${_dir} NAME)
    string(TOUPPER ${_ARG_IDENTIFIERS}_${_dir_name} _dir_identifier)
    string(REPLACE "." "DOT" _dir_identifier ${_dir_identifier})

    # test if no path is specified
    if(NOT ${_dir_name} STREQUAL ${_dir})
      message(WARNING "Argument DIRECTORIES_${_assembler_upper} only checks for directory name for the given IDENTIFIER. "
        "Appended paths to the directory name will be ignored.")
    endif()

    # check if dir identifier is registered
    if(NOT ${_dir_identifier} IN_LIST _vid_registered_inc_dirs_assembler)
      message(FATAL_ERROR "${_dir} is not registered for identifier ${_ARG_IDENTIFIERS}")
    endif()

    # append created dir identifier to a list of dir identifiers
    list(APPEND _list_directories_assembler ${_dir_identifier})
  endforeach(_dir)

  # argument IDENTIFIER and no directories for the assembler are specified, add dirs as includes
  # iterate over identifiers
  foreach(_identifier ${_ARG_IDENTIFIERS})
    # set current identifier with upper case
    string(TOUPPER ${_identifier} _identifier_upper)

    # check only for compiler include dirs when no assembler DIRECTORIES are specified
    if(NOT DEFINED _ARG_DIRECTORIES_${_assembler_upper})
      # use reverse index to get only dirs registered under this identifier (O(k) vs O(N) scan)
      get_property(_identifier_dirs_compiler GLOBAL PROPERTY VID_REGISTERED_INC_DIRS_${_compiler_upper}_ID_${_identifier_upper})
      foreach(_dir ${_identifier_dirs_compiler})
        # get full path of registered compiler include dir
        get_property(_dir_path GLOBAL PROPERTY ${_dir}_${_compiler_upper}_PATH)

        # check if only specific dirs should be added
        if(DEFINED _ARG_DIRECTORIES)
          # check if registered include dir is in list of needed include dirs
          if(${_dir} IN_LIST _list_directories)
            # append the dir to list of includes
            list(APPEND VID_TARGET_${_target_upper}_${_compiler_upper}_INCLUDES ${_dir_path})
          endif()
        # all dirs of an identifier should be added
        else()
          # append the dir to list of includes
          list(APPEND VID_TARGET_${_target_upper}_${_compiler_upper}_INCLUDES ${_dir_path})
        endif()
      endforeach(_dir)
    endif()

    # check only for assembler include dirs when no compiler DIRECTORIES are specified
    if(NOT DEFINED _ARG_DIRECTORIES)
      # use reverse index to get only dirs registered under this identifier (O(k) vs O(N) scan)
      get_property(_identifier_dirs_assembler GLOBAL PROPERTY VID_REGISTERED_INC_DIRS_${_assembler_upper}_ID_${_identifier_upper})
      foreach(_dir ${_identifier_dirs_assembler})
        # get full path of registered assembler include dir
        get_property(_dir_path GLOBAL PROPERTY ${_dir}_${_assembler_upper}_PATH)

        # check if only specific dirs should be added
        if(DEFINED _ARG_DIRECTORIES_${_assembler_upper})
          # check if registered include dir is in list of needed include dirs
          if(${_dir} IN_LIST _list_directories_assembler)
            # append the dir to list of includes
            list(APPEND VID_TARGET_${_target_upper}_${_assembler_upper}_INCLUDES ${_dir_path})
          endif()
        # all dirs of an identifier should be added
        else()
          # append the dir to list of includes
          list(APPEND VID_TARGET_${_target_upper}_${_assembler_upper}_INCLUDES ${_dir_path})
        endif()
      endforeach(_dir)
    endif()
  endforeach(_identifier)
endmacro(_add_includes)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_includes

  Macro for adding include directories to a videantis build target.
  This macro requires at least one IDENTIFIER to add include dirs
  or one directory added as local directory via ADD or ADD_VMPASM.
  If not all include dirs of one IDENTIFIER should be added,
  the directories can be selected with DIRECTORIES/DIRECTORIES_VMPASM.

  .. code-block:: cmake

    vid_add_includes(target [IDENTIFIERS id1 [id2 ...]] [DIRECTORIES dir1 [dir2 ...]]
        [DIRECTORIES_VMPASM dir1 [dir2 ...]] [ADD dir1 [dir2 ...]]
        [ADD_VMPASM dir1 [dir2 ...]] [PRINT_AVAILABLE_INCLUDES])

  Function parameters:

  ``target``
    name of an already defined build target
  ``IDENTIFIERS``
    list of identifiers for include dirs
  ``DIRECTORIES``
    list of dirs from the identifier (no full path)
  ``DIRECTORIES_VMPASM``
    list of dirs from the identifier (no full path)

  .. note::

    Only one identifier is allowed when DIRECTORIES/DIRECTORIES_VMPASM is set (:cmake:command:`vid_add_includes` can be called multiple times).
    If an include directory is registered as . (dot), the character . (dot) has to be used to select it.

  ``ADD``
    list of dirs add as includes without registering as globally available dir
  ``ADD_VMPASM``
    list of dirs add as includes for vmpasm without registering as globally available dir
  ``PRINT_AVAILABLE_INCLUDES``
    print globally available includes

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VMPCC_INCLUDES``
    list of include directories for vmpcc with full path
  ``VID_TARGET_<TARGET_NAME>_VMPASM_INCLUDES``
    list of include directories for vmpasm with full path

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
#]=======================================================================]
macro(vid_add_includes _target)
  cmake_parse_arguments(_ARG "PRINT_AVAILABLE_INCLUDES" ""
    "IDENTIFIERS;DIRECTORIES;DIRECTORIES_VMPASM;ADD;ADD_VMPASM" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_add_includes() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VMP")
    message(FATAL_ERROR "vid_add_includes() needs to be called for a v-MP target")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_includes() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # get length of lists for IDENTIFIERS
  list(LENGTH _ARG_IDENTIFIERS _num_identifiers)

  # do error handling when calling this macro pending on input arguments
  # setting IDENTIFIERS is required
  if((${_num_identifiers} EQUAL 0) AND (NOT DEFINED _ARG_ADD) AND (NOT DEFINED _ARG_ADD_VMPASM))
    message(FATAL_ERROR "vid_add_includes() requires minimum one IDENTIFIER or one dir via ADD/ADD_VMPASM")
  endif()
  # when paths are selected with DIRECTORIES only one IDENTIFIER is allowed
  if((${_num_identifiers} GREATER 1) AND ((DEFINED _ARG_DIRECTORIES) OR (DEFINED _ARG_DIRECTORIES_VMPASM)))
    message(FATAL_ERROR "vid_add_includes() can take only multiple IDENTIFIERS, when no DIRECTORIES/DIRECTORIES_VMPASM paths are specified")
  endif()
  if(((DEFINED _ARG_DIRECTORIES) OR (DEFINED _ARG_DIRECTORIES_VMPASM)) AND (${_num_identifiers} EQUAL 0))
    message(FATAL_ERROR "vid_add_includes() needs one IDENTIFIER, when DIRECTORIES/DIRECTORIES_VMPASM paths are specified")
  endif()

  # unset print_available_includes from possible prior calls of vid_add_includes()
  unset(_print_available_includes)
  # check if PRINT_AVAILABLE_INCLUDES is set
  if(${_ARG_PRINT_AVAILABLE_INCLUDES})
    set(_print_available_includes PRINT_AVAILABLE_INCLUDES)
  endif()

  # call internal macro to add includes
  _add_includes(${_target} vmpcc vmpasm IDENTIFIERS ${_ARG_IDENTIFIERS} DIRECTORIES ${_ARG_DIRECTORIES}
    DIRECTORIES_VMPASM ${_ARG_DIRECTORIES_VMPASM} ADD ${_ARG_ADD} ADD_VMPASM ${_ARG_ADD_VMPASM} ${_print_available_includes})
endmacro(vid_add_includes)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_includes_vmp

  Macro for adding include directories to a videantis v-MP build target.
  This is a wrapper macro for :cmake:command:`vid_add_includes`.

  .. code-block:: cmake

    vid_add_includes_vmp(target [IDENTIFIERS id1 [id2 ...]] [DIRECTORIES dir1 [dir2 ...]]
        [DIRECTORIES_VMPASM dir1 [dir2 ...]] [ADD dir1 [dir2 ...]]
        [ADD_VMPASM dir1 [dir2 ...]] [PRINT_AVAILABLE_INCLUDES])

  Function parameters:

  ``target``
    name of an already defined build target
  ``IDENTIFIERS``
    list of identifiers for include dirs
  ``DIRECTORIES``
    list of dirs from the identifier (no full path)
  ``DIRECTORIES_VMPASM``
    list of dirs from the identifier (no full path)

  .. note:

    Only one identifier is allow when DIRECTORIES/DIRECTORIES_VMPASM is set (:cmake:command:`vid_add_includes_vmp` can be called multiple times).
    If an include directory is registered as . (dot), the character . (dot) has to be used to select it.

  ``ADD``
    list of dirs add as includes without registering as globally available dir
  ``ADD_VMPASM``
    list of dirs add as includes for vmpasm without registering as globally available dir
  ``PRINT_AVAILABLE_INCLUDES``
    print globally available includes
#]=======================================================================]
macro(vid_add_includes_vmp _target)
  cmake_parse_arguments(_ARG "PRINT_AVAILABLE_INCLUDES" ""
    "IDENTIFIERS;DIRECTORIES;DIRECTORIES_VMPASM;ADD;ADD_VMPASM" ${ARGN})

  # unset print_available_includes from possible prior calls of vid_add_includes_vmp()
  unset(_print_available_includes)
  # check if PRINT_AVAILABLE_INCLUDES is set
  if(${_ARG_PRINT_AVAILABLE_INCLUDES})
    set(_print_available_includes PRINT_AVAILABLE_INCLUDES)
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_includes_vmp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  vid_add_includes(${_target} IDENTIFIERS ${_ARG_IDENTIFIERS} DIRECTORIES ${_ARG_DIRECTORIES}
    DIRECTORIES_VMPASM ${_ARG_DIRECTORIES_VMPASM} ADD ${_ARG_ADD} ADD_VMPASM ${_ARG_ADD_VMPASM} ${_print_available_includes})
endmacro(vid_add_includes_vmp)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_defines

  Macro for adding defines to a videantis build target.
  The define needs to be specified without "-D" in the beginning
  and is not allowed to contain whitespace.

  .. code-block:: cmake

    vid_add_defines(target [ALL define1 [define2 ...]]
        [VMPCC define1 [define2 ...]] [VMPASM define1 [define2 ...]])

  Function parameters:

  ``target``
    name of an already defined build target
  ``ALL``
    add one or more defines to vmpcc and vmpasm
  ``VMPCC``
    add one or more defines to vmpcc
  ``VMPASM``
    add one or more defines to vmpasm

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VMPCC_DEFINES``
    list of defines for vmpcc
  ``VID_TARGET_<TARGET_NAME>_VMPASM_DEFINES``
    list of defines for vmpasm

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
#]=======================================================================]
macro(vid_add_defines _target)
  cmake_parse_arguments(_ARG "" ""
    "ALL;VMPCC;VMPASM" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_add_defines() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VMP")
    message(FATAL_ERROR "vid_add_defines() needs to be called for a v-MP target")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_defines() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # check if at least one argument (ALL, VMPCC or VMPASM) with value is present in the macro call
  if((NOT DEFINED _ARG_ALL) AND (NOT DEFINED _ARG_VMPCC) AND (NOT DEFINED _ARG_VMPASM))
    message(FATAL_ERROR "Argument (ALL, VMPCC or VMPASM) and/or value is missing")
  endif()

  # append defines to vmpcc and vmpasm define lists
  list(APPEND VID_TARGET_${_target_upper}_VMPCC_DEFINES ${_ARG_ALL})
  list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES ${_ARG_ALL})

  # append defines to vmpcc define list
  list(APPEND VID_TARGET_${_target_upper}_VMPCC_DEFINES ${_ARG_VMPCC})

  # append defines to vmpasm define list
  list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES ${_ARG_VMPASM})
endmacro(vid_add_defines)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_defines_vmp

  Macro for adding defines to a videantis v-MP build target.
  This is a wrapper macro for :cmake:command:`vid_add_defines`.

  .. code-block:: cmake

    vid_add_defines_vmp(target [ALL define1 [define2 ...]]
        [VMPCC define1 [define2 ...]] [VMPASM define1 [define2 ...]])

  Function parameters:

  ``target``
    name of an already defined build target
  ``ALL``
    add one or more defines to vmpcc and vmpasm
  ``VMPCC``
    add one or more defines to vmpcc
  ``VMPASM``
    add one or more defines to vmpasm
#]=======================================================================]
macro(vid_add_defines_vmp _target)
  cmake_parse_arguments(_ARG "" ""
    "ALL;VMPCC;VMPASM" ${ARGN})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_defines_vmp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  vid_add_defines(${_target} ALL ${_ARG_ALL} VMPCC ${_ARG_VMPCC} VMPASM ${_ARG_VMPASM})
endmacro(vid_add_defines_vmp)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_compile_flags

  Macro for adding vmpcc flags to a videantis build target.
  The flags should contain no whitespace.

  .. code-block:: cmake

    vid_add_compile_flags(target FLAGS flag1 [flag2 ...])

  Function parameters:

  ``target``
    name of an already defined build target
  ``FLAGS``
    add one or more flags

  Exposed variables:

  ``VID_TARGET_<TARGET_NAME>_VMPCC_FLAGS``
    list of flags for vmpcc

  Placeholder:

  ``<TARGET_NAME>``
    target name in all upper case letters
#]=======================================================================]
macro(vid_add_compile_flags _target)
  cmake_parse_arguments(_ARG "" ""
    "FLAGS" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_add_compile_flags() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VMP")
    message(FATAL_ERROR "vid_add_compile_flags() needs to be called for a v-MP target")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_compile_flags() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # check if FLAGS is present in the macro call
  if(NOT DEFINED _ARG_FLAGS)
    message(FATAL_ERROR "Argument FLAGS is mandatory for macro vid_add_compile_flags()")
  endif()

  # append flags to vmpcc flags list
  list(APPEND VID_TARGET_${_target_upper}_VMPCC_FLAGS ${_ARG_FLAGS})
endmacro(vid_add_compile_flags)

#[=======================================================================[.rst:
.. cmake:command:: vid_add_compile_flags_vmp

  Macro for adding vmpcc flags to a videantis v-MP build target.
  This is a wrapper macro for :cmake:command:`vid_add_compile_flags`.

  .. code-block:: cmake

    vid_add_compile_flags_vmp(target FLAGS flag1 [flag2 ...])

  Function parameters:

  ``target``
    name of an already defined build target
  ``FLAGS``
    add one or more flags
#]=======================================================================]
macro(vid_add_compile_flags_vmp _target)
  cmake_parse_arguments(_ARG "" ""
    "FLAGS" ${ARGN})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_add_compile_flags_vmp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  vid_add_compile_flags(${_target} FLAGS ${_ARG_FLAGS})
endmacro(vid_add_compile_flags_vmp)

#[=======================================================================[.rst:
.. cmake:command:: vid_build_target

  Macro for building a videantis build target.

  The cmake module defines for every build target a define for assembler
  and compiler with the target cpu. E.g. v-MP 4.0 target cpu creates the
  define `__mp400__`.

  .. code-block:: cmake

    vid_build_target(target [DEPENDENCIES dependency1 [dependency2 ...]]
        [WORKING_DIRECTORY dir] [BIN2H] [NICEOUT] [NO_TARGET] [NO_ALL])

  Function parameters:

  ``target``
    name of an already defined build target
  ``DEPENDENCIES``
    add one or more cmake dependencies
  ``WORKING_DIRECTORY``
    define working directory (default: current cmake binary directory)
  ``BIN2H``
    create header file with binary code
  ``NICEOUT``
    create map file with --out-nice-memory-imem vspasm option output
  ``NO_TARGET``
    create no cmake targets
  ``NO_ALL``
    build target won't be added to ALL target of the build

  Exposed global variables:

  ``VID_TARGET_<TARGET_NAME>_WORKING_DIR``
    full path of the working directory of the build target
  ``VID_TARGET_<TARGET_NAME>_VMPCC_SOURCES``
    list of vmpcc sources of the build target
  ``VID_TARGET_<TARGET_NAME>_VMPASM_SOURCES``
    list of vmpasm sources of the build target
  ``VID_TARGET_<TARGET_NAME>_SOURCES``
    combined list of all sources of the build target
  ``VID_TARGET_<TARGET_NAME>_VMPCC_OUTPUTS``
    list of vmpcc outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_VMPASM_OUTPUTS``
    list of vmpasm outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_BINARY_OUTPUTS``
    list of binary outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_BIN2H_OUTPUTS``
    list of bin2h outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_OUTPUTS``
    combined list of all outputs of the build target
  ``VID_TARGET_<TARGET_NAME>_VMPCC_COMMANDS``
    list of vmpcc commands of the build target
  ``VID_TARGET_<TARGET_NAME>_VMPASM_COMMANDS``
    list of vmpasm commands of the build target
  ``VID_TARGET_<TARGET_NAME>_BINARY_COMMANDS``
    list of binary commands of the build target
  ``VID_TARGET_<TARGET_NAME>_BIN2H_COMMANDS``
    list of bin2h commands of the build target
  ``VID_TARGET_<TARGET_NAME>_COMMANDS``
    combined list of all commands of the build target

  Exposed global properties:

  ``VID_TARGET_<TARGET_NAME>_BUILD``
    set to true when :cmake:command:`vid_build_target` is called
  ``VID_TARGET_<TARGET_NAME>_BIN_FILE``
    generated bin file (full path) for target
  ``VID_TARGET_<TARGET_NAME>_BIN2H_FILE``
    generated bin2h file (full path) for target
  ``VID_TARGET_<TARGET_NAME>_MAP_FILE``
    generated map file (full path) for target

  Exposed cmake targets:

  ``compile_<vmpcc_output_name>``
    compile only one source file
  ``build_<target_name>``
    build target

  Placeholder:

  ``<target_name>``
    target name
  ``<TARGET_NAME>``
    target name in all upper case letters
  ``<vmpcc_output_name>``
    <target_name>_[<identifier>_]<source_filename_we>
  ``<identifier>``
    identifier if file was globally added with :cmake:command:`vid_register_file`
  ``<source_filename_we>``
    vmpcc source file name without file extension
#]=======================================================================]
macro(vid_build_target _target)
  cmake_parse_arguments(_ARG "BIN2H;NICEOUT;NO_TARGET;NO_ALL" "WORKING_DIRECTORY"
    "DEPENDENCIES" ${ARGN})

  # set target with upper case
  string(TOUPPER ${_target} _target_upper)
  # check if target for this videantis build process is available
  get_property(_target_property GLOBAL PROPERTY VID_TARGET_${_target_upper})
  if(NOT "${_target_property}" STREQUAL ${_target})
    message(FATAL_ERROR "Target ${_target} is not found in videantis build context")
  endif()

  # check if vid_build_target() is already called for this target
  get_property(_target_build GLOBAL PROPERTY VID_TARGET_${_target_upper}_BUILD)
  if(DEFINED _target_build)
    message(FATAL_ERROR "vid_build_target() has been already called for target ${_target}")
  endif()
  # set this target as build, if vid_build_target() was not called before
  set_property(GLOBAL PROPERTY VID_TARGET_${_target_upper}_BUILD TRUE)

  # check if variable for this target is found (ensures macro are called in the correct scope)
  if(NOT DEFINED VID_TARGET_${_target_upper})
    message(FATAL_ERROR "Variable VID_TARGET_${_target_upper} not found. vid_build_target() might be called in the wrong scope")
  endif()

  # get core type of target
  get_property(_type GLOBAL PROPERTY VID_TARGET_${_target_upper}_TYPE)
  # check if target has correct core type
  if(NOT ${_type} STREQUAL "VMP")
    message(FATAL_ERROR "vid_build_target() needs to be called for a v-MP target")
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
    message(WARNING "vid_build_target() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
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

  # clear internal variables from previous calls of vid_build_target()
  unset(_vmpcc_include_flags)
  unset(_vmpcc_define_flags)
  unset(_vmpasm_include_flags)
  unset(_vmpasm_define_flags)
  unset(_vmpasm_out_files)
  unset(_vmpasm_mandatory_out_files)
  unset(_bin_input_files)
  unset(_bin2h_file)
  unset(_vmp_lib_files)
  unset(_vmp_4x_defines)

  # get number of list elements for vmpcc sources and outputs
  list(LENGTH VID_TARGET_${_target_upper}_VMPCC_SOURCES _num_vmpcc_sources)
  list(LENGTH VID_TARGET_${_target_upper}_VMPCC_OUTPUTS _num_vmpcc_outputs)
  # check if the number of vmpcc sources and outputs match
  if(NOT ${_num_vmpcc_sources} EQUAL ${_num_vmpcc_outputs})
    message(FATAL_ERROR "Number of sources and outputs for vmpcc do not match, something went wrong while setting up the target")
  endif()

  # remove possible duplicates in lists of vmpcc sources and duplicates
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VMPCC_SOURCES)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VMPCC_OUTPUTS)
  # get number of list elements for vmpcc sources and outputs after removing possible duplicates
  list(LENGTH VID_TARGET_${_target_upper}_VMPCC_SOURCES _num_vmpcc_sources)
  list(LENGTH VID_TARGET_${_target_upper}_VMPCC_OUTPUTS _num_vmpcc_outputs)
  # check if the number of vmpcc sources and outputs match after removing possible duplicates
  if(NOT ${_num_vmpcc_sources} EQUAL ${_num_vmpcc_outputs})
    message(FATAL_ERROR "Number of sources and outputs for vmpcc do not match after removing duplicates. "
      "It could be possible that a file was added twice. Globally and locally or with different identifiers")
  endif()

  # remove possible duplicates from vmpasm sources and includes of vmpcc and vmpasm
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VMPASM_SOURCES)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VMPCC_INCLUDES)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VMPASM_INCLUDES)

  # create target cpu define
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _ ${VID_TARGET_${_target_upper}_MP_VERSION})
  set(_mp_version_major ${CMAKE_MATCH_1})
  set(_mp_version_minor ${CMAKE_MATCH_2})
  if(${_mp_version_minor} LESS 10)
    set(_mp_version_minor 0${_mp_version_minor})
  endif()
  set(_target_cpu_define __core_version__=${_mp_version_major}${_mp_version_minor})
  # add target cpu as define
  list(APPEND VID_TARGET_${_target_upper}_VMPCC_DEFINES ${_target_cpu_define})
  list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES ${_target_cpu_define})

  # check if some lowlevel library defines should be overwritten
  # check if VMP_LLVM_STACK_SIZE (size of llvm stack) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_LLVM_STACK_SIZE)
    # bootloader has changed during lowlevel library 3.0 development for v-MP version greater equal 4.0
    # assembler variable is named VMP_LLVM_STACK_SIZE (VID_VMP_LLVM_STACK_SIZE is deprecated)
    if(${VID_TARGET_${_target_upper}_MP_VERSION} VERSION_GREATER_EQUAL "4.0")
      # append to assembler defines
      list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_LLVM_STACK_SIZE=${VID_TARGET_${_target_upper}_VMP_LLVM_STACK_SIZE})
    else()
      # append to assembler defines
      list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VID_VMP_LLVM_STACK_SIZE=${VID_TARGET_${_target_upper}_VMP_LLVM_STACK_SIZE})
    endif()
  endif()
  # check if VMP_LLVM_STACK_MEM (placement memory for llvm stack) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_LLVM_STACK_MEM)
    # check and translate data memory
    vid_check_and_translate_data_memory_vmp(VID_TARGET_${_target_upper}_VMP_LLVM_STACK_MEM)
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_LLVM_STACK_MEM=${_DATA_MEMORY_ID})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_LLVM_STACK_MEM)
  endif()
  # check if VMP_LLVM_STACK_ORG (placement location for llvm stack) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_LLVM_STACK_ORG)
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_LLVM_STACK_ORG=${VID_TARGET_${_target_upper}_VMP_LLVM_STACK_ORG})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_LLVM_STACK_ORG)
  endif()
  # check if VMP_MBOX_MEM (placement memory for mbox segment) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_MBOX_MEM)
    # check and translate data memory
    vid_check_and_translate_data_memory_vmp(VID_TARGET_${_target_upper}_VMP_MBOX_MEM)
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_MBOX_MEM=${_DATA_MEMORY_ID})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_MBOX_MEM)
  endif()
  # check if VMP_OLM_MEM (placement memory for overlay manager segment) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_OLM_MEM)
    # check and translate data memory
    vid_check_and_translate_data_memory_vmp(VID_TARGET_${_target_upper}_VMP_OLM_MEM)
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_OLM_MEM=${_DATA_MEMORY_ID})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_OLM_MEM)
  endif()
  # check if VMP_NUM_DMA_CHANNELS (number of DMA descriptors generated) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_NUM_DMA_CHANNELS)
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_NUM_DMA_CHANNELS=${VID_TARGET_${_target_upper}_VMP_NUM_DMA_CHANNELS})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_NUM_DMA_CHANNELS)
  endif()
  # check if VMP_DMA_DESCR_MEM (placement memory for DMA descriptor segment) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_DMA_DESCR_MEM)
    # check and translate data memory
    vid_check_and_translate_data_memory_vmp(VID_TARGET_${_target_upper}_VMP_DMA_DESCR_MEM)
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_DMA_DESCR_MEM=${_DATA_MEMORY_ID})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_DMA_DESCR_MEM)
  endif()
  # check if VMP_DMA_DESCR_ORG (placement location for DMA descriptor segment) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_DMA_DESCR_ORG)
    # append to compiler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPCC_DEFINES VMP_DMA_DESCR_ORG=${VID_TARGET_${_target_upper}_VMP_DMA_DESCR_ORG})
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_DMA_DESCR_ORG=${VID_TARGET_${_target_upper}_VMP_DMA_DESCR_ORG})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_DMA_DESCR_ORG)
  endif()
  # check if VMP_DMA_LEGACY_ORG (placement location for legacy DMA segment) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_DMA_LEGACY_ORG)
    # append to compiler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPCC_DEFINES VMP_DMA_LEGACY_ORG=${VID_TARGET_${_target_upper}_VMP_DMA_LEGACY_ORG})
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_DMA_LEGACY_ORG=${VID_TARGET_${_target_upper}_VMP_DMA_LEGACY_ORG})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_DMA_LEGACY_ORG)
  endif()
  # check if VMP_DMA_LEGACY_HIGH (default upper address 32 bits for legacy transfers) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_DMA_LEGACY_HIGH)
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_DMA_LEGACY_HIGH=${VID_TARGET_${_target_upper}_VMP_DMA_LEGACY_HIGH})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_DMA_LEGACY_HIGH)
  endif()
  # check if VMP_EDMA_DESCR_MEM (placement memory for EDMA descriptor segment) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_EDMA_DESCR_MEM)
    # check and translate data memory
    vid_check_and_translate_data_memory_vmp(VID_TARGET_${_target_upper}_VMP_EDMA_DESCR_MEM)
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_EDMA_DESCR_MEM=${_DATA_MEMORY_ID})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_EDMA_DESCR_MEM)
  endif()
  # check if VMP_EDMA_DESCR_ORG (placement location for EDMA descriptor segment) is defined to overwrite lowlevel library default
  if(DEFINED VID_TARGET_${_target_upper}_VMP_EDMA_DESCR_ORG)
    # append to compiler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPCC_DEFINES VMP_EDMA_DESCR_ORG=${VID_TARGET_${_target_upper}_VMP_EDMA_DESCR_ORG})
    # append to assembler defines
    list(APPEND VID_TARGET_${_target_upper}_VMPASM_DEFINES VMP_EDMA_DESCR_ORG=${VID_TARGET_${_target_upper}_VMP_EDMA_DESCR_ORG})
    # mark that a define for v-MP 4.0 and greater is present
    list(APPEND _vmp_4x_defines VID_TARGET_${_target_upper}_VMP_EDMA_DESCR_ORG)
  endif()

  # check if defines for v-MP 4.0 and greater were present for older v-MP revisions
  if((DEFINED _vmp_4x_defines) AND (${VID_TARGET_${_target_upper}_MP_VERSION} VERSION_LESS "4.0"))
    # transform cmake list into string list
    string(REPLACE ";" ", " _vmp_4x_defines "${_vmp_4x_defines}")
    # raise warning
    message(WARNING "The following variables are only supported for v-MP 4.0 and greater and will be ignored: ${_vmp_4x_defines}")
  endif()

  # remove possible duplicates from vmpcc flags and defines of vmpcc and vmpasm
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VMPCC_FLAGS)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VMPCC_DEFINES)
  list(REMOVE_DUPLICATES VID_TARGET_${_target_upper}_VMPASM_DEFINES)

  # iterate over vmpcc includes and add -I
  foreach(_vmpcc_include ${VID_TARGET_${_target_upper}_VMPCC_INCLUDES})
    list(APPEND _vmpcc_include_flags "-I${_vmpcc_include}")
  endforeach(_vmpcc_include)

  # iterate over vmpcc defines and add -D
  foreach(_vmpcc_define ${VID_TARGET_${_target_upper}_VMPCC_DEFINES})
    list(APPEND _vmpcc_define_flags "-D${_vmpcc_define}")
  endforeach(_vmpcc_define)

  # create a combined list of all vmpcc flags, includes and defines
  set(_vmpcc_flags ${VID_TARGET_${_target_upper}_VMPCC_FLAGS}
                   ${VID_TARGET_${_target_upper}_VMPCC_TARGET_FLAGS}
                   ${VID_TARGET_${_target_upper}_VMPCC_OPTIMIZATION_LEVEL}
                   ${_vmpcc_include_flags}
                   ${_vmpcc_define_flags})

  # iterate over all vmpcc sources to create custom commands to compile every source
  foreach(_vmpcc_source _vmpcc_output IN ZIP_LISTS VID_TARGET_${_target_upper}_VMPCC_SOURCES VID_TARGET_${_target_upper}_VMPCC_OUTPUTS)
    # get output name without extension
    get_filename_component(_output_filename ${_vmpcc_output} NAME_WLE)

    # get source file filename
    get_filename_component(_source_filename ${_vmpcc_source} NAME)

    # get output filename as all upper case letters string
    string(TOUPPER ${_output_filename} _output_filename_upper)

    # unset vmpcc optional flags and outputs from previous iterations or macro calls
    unset(_vmpcc_optional_flags)
    unset(_vmpcc_optional_outputs)

    # check if a vidasm helper file should be created
    if(${VID_TARGET_${_output_filename_upper}_VIDASM_HELPER_FILE})
      # define vidasm helper filename
      set(_vidasm_helper_filename ${_output_filename}_vidasm_helper.asm)

      # append helper file flag to optional flags
      list(APPEND _vmpcc_optional_flags -vidasm-helper-file=${_vidasm_helper_filename})
      # append helper file output file to optional outputs
      list(APPEND _vmpcc_optional_outputs ${_working_dir}/${_vidasm_helper_filename})

      # append vidasm helper file to vmpcc outputs (input for vmpasm)
      list(APPEND VID_TARGET_${_target_upper}_VMPCC_OUTPUTS ${_vidasm_helper_filename})
    endif()

    # get length of vmpcc commands list
    list(LENGTH VID_TARGET_${_target_upper}_VMPCC_COMMANDS _vmpcc_cmd_length)
    # if list of vmpcc commands contains something add && to list
    if(${_vmpcc_cmd_length} GREATER "0")
      list(APPEND VID_TARGET_${_target_upper}_VMPCC_COMMANDS &&)
    endif()
    # add current vmpcc command to vmpcc commands list
    list(APPEND VID_TARGET_${_target_upper}_VMPCC_COMMANDS ${VMPCC_EXECUTABLE} ${_vmpcc_flags} ${_vmpcc_optional_flags} -o ${_vmpcc_output} ${_vmpcc_source})

    # check if a target should be created
    if(NOT ${_ARG_NO_TARGET})
      # compile source with vmpcc
      add_custom_command(OUTPUT ${_working_dir}/${_vmpcc_output} ${_working_dir}/${_vmpcc_output}.d ${_vmpcc_optional_outputs}
        COMMAND ${VMPCC_EXECUTABLE} ${_vmpcc_flags} ${_vmpcc_optional_flags} -MD -MT ${_vmpcc_output} -MF ${_vmpcc_output}.d -o ${_vmpcc_output} ${_vmpcc_source}
        DEPENDS ${_vmpcc_source}
        DEPFILE ${_working_dir}/${_vmpcc_output}.d
        WORKING_DIRECTORY ${_working_dir}
        VERBATIM COMMENT "Compiling ${_source_filename} to ${_vmpcc_output} ...")

      # add custom target for current output
      # this enables to run vmpcc only for one source (developing or debugging process)
      add_custom_target(compile_${_output_filename}
        DEPENDS ${_working_dir}/${_vmpcc_output})
    endif()
  endforeach(_vmpcc_source)

  # iterate over vmpasm includes and add --include-dir=
  foreach(_vmpasm_include ${VID_TARGET_${_target_upper}_VMPASM_INCLUDES})
    list(APPEND _vmpasm_include_flags "--include-dir=${_vmpasm_include}")
  endforeach(_vmpasm_include)

  # iterate over vmpasm defines and add -D
  foreach(_vmpasm_define ${VID_TARGET_${_target_upper}_VMPASM_DEFINES})
    list(APPEND _vmpasm_define_flags "-D${_vmpasm_define}")
  endforeach(_vmpasm_define)

  # set --size-memory-dmemX flags depending on the defined DMEM size
  set(_vmpasm_memory_flags "--size-memory-dmem=${VID_TARGET_${_target_upper}_SIZE_MEMORY_DMEM}"
                           "--size-memory-dmem2=${VID_TARGET_${_target_upper}_SIZE_MEMORY_DMEM2}")
  # if mp version is greater equal 4.0 append dmem3 memory size flag
  if(${VID_TARGET_${_target_upper}_MP_VERSION} VERSION_GREATER_EQUAL "4.0")
    list(APPEND _vmpasm_memory_flags "--size-memory-dmem3=${VID_TARGET_${_target_upper}_SIZE_MEMORY_DMEM3}")
  endif()

  # iterate over all vmpasm out flags
  foreach(_vmpasm_out_flag ${VID_TARGET_${_target_upper}_VMPASM_OUT_FLAGS})
    # strip flag to ensure no whitespace are at beginning or end of the flag
    string(STRIP ${_vmpasm_out_flag} _vmpasm_out_flag)
    # extract output filename from the flag
    string(REGEX MATCH "=([a-zA-Z0-9_\\.\\-]+)$" _ ${_vmpasm_out_flag} )
    set(_vmpasm_out_file ${CMAKE_MATCH_1})

    # remove extension from output filename
    get_filename_component(_vmpasm_out_filename ${_vmpasm_out_file} NAME_WE)
    # check that output filename without extension is the target (naming convention for easier processing)
    if(NOT ${_vmpasm_out_filename} STREQUAL ${_target})
      message(FATAL_ERROR "vmpasm output file (${_out_file}) does not apply to naming convention: <target>.extension")
    endif()

    # append output file to list of output files
    list(APPEND _vmpasm_out_files ${_vmpasm_out_file})
  endforeach(_vmpasm_out_flag)

  # define mandatory vmpasm output files and append them to a list
  set(_vmpasm_memory_map_file ${_target}.nocodemap)
  list(APPEND _vmpasm_mandatory_out_files ${_vmpasm_memory_map_file})
  set(_vmpasm_nice_memory_imem_file ${_target}.niceout.asm)
  list(APPEND _vmpasm_mandatory_out_files ${_vmpasm_nice_memory_imem_file})
  set(_vmpasm_hw_asm_file ${_target}.hw.asm)
  list(APPEND _vmpasm_mandatory_out_files ${_vmpasm_hw_asm_file})
  set(_vmpasm_code_file ${_target}.imem)
  list(APPEND _vmpasm_mandatory_out_files ${_vmpasm_code_file})
  list(APPEND _bin_input_files ${_vmpasm_code_file})
  set(_vmpasm_memory_code_ext_file ${_target}.ovl.imem)
  list(APPEND _vmpasm_mandatory_out_files ${_vmpasm_memory_code_ext_file})
  list(APPEND _bin_input_files ${_vmpasm_memory_code_ext_file})
  set(_vmpasm_memory_dmem_file ${_target}.dmem)
  list(APPEND _vmpasm_mandatory_out_files ${_vmpasm_memory_dmem_file})
  list(APPEND _bin_input_files ${_vmpasm_memory_dmem_file})
  set(_vmpasm_memory_dmem2_file ${_target}.dmem2)
  list(APPEND _vmpasm_mandatory_out_files ${_vmpasm_memory_dmem2_file})
  list(APPEND _bin_input_files ${_vmpasm_memory_dmem2_file})
  if(${VID_TARGET_${_target_upper}_MP_VERSION} VERSION_GREATER_EQUAL "4.0")
    set(_vmpasm_memory_dmem3_file ${_target}.dmem3)
    list(APPEND _vmpasm_mandatory_out_files ${_vmpasm_memory_dmem3_file})
    list(APPEND _bin_input_files ${_vmpasm_memory_dmem3_file})
  endif()

  # check if all mandatory output files are in the list of vmpasm output files
  foreach(_vmpasm_mandatory_out_file ${_vmpasm_mandatory_out_files})
    if(NOT ${_vmpasm_mandatory_out_file} IN_LIST _vmpasm_out_files)
      message(FATAL_ERROR "File ${_vmpasm_mandatory_out_file} is not in the list of vmpasm outputs. "
        "Check VID_TARGET_${_target_upper}_VMPASM_OUT_FLAGS cmake variable!")
    endif()
  endforeach(_vmpasm_mandatory_out_file)

  # create a combined list of all vmpasm flags, includes and defines
  set(_vmpasm_flags ${VID_TARGET_${_target_upper}_VMPASM_TARGET_FLAGS}
                    ${VID_TARGET_${_target_upper}_VMPASM_STATISTIC_FLAGS}
                    ${VID_TARGET_${_target_upper}_VMPASM_OLM_BOOTADDR_FLAGS}
                    ${VID_TARGET_${_target_upper}_VMPASM_HW_SOURCECODE_INFO_FLAGS}
                    ${_vmpasm_include_flags}
                    ${_vmpasm_define_flags}
                    ${_vmpasm_memory_flags}
                    ${VID_TARGET_${_target_upper}_VMPASM_OUT_FLAGS}
                    ${VID_TARGET_${_target_upper}_VMPASM_CUSTOM_FLAGS}
                    ${VID_TARGET_${_target_upper}_VMPASM_DONT_ELIMINATE_DMA_LEGACY}
                    ${VID_TARGET_${_target_upper}_VMPASM_DONT_ELIMINATE_EDMA_DESCR})

  # iterate over all lowlevel library files specified
  foreach(_vmp_lib_file ${VID_TARGET_${_target_upper}_VMP_LIB_FILES})
    # generate full path for lllib file based on LLLIB v-MP lib dir (defined in lllib cmake file)
    string(JOIN "/" _vmp_lib_file_full_path ${VID_TARGET_${_target_upper}_LLLIB_VMP_LIB_DIR} ${_vmp_lib_file})

    # check if file exists
    if(NOT EXISTS ${_vmp_lib_file_full_path})
      message(FATAL_ERROR "${_vmp_lib_file} does not exists in ${VID_TARGET_${_target_upper}_LLLIB_VMP_LIB_DIR}")
    endif()

    # append file to list with full path lllib files for dependencies of assembling step
    list(APPEND _vmp_lib_files ${_vmp_lib_file_full_path})
  endforeach(_vmp_lib_file)

  # create a combined list of all vmpasm sources
  # (assembler libraries, compiled cl-code and assembler sources)
  set(_vmpasm_sources ${VID_TARGET_${_target_upper}_VMP_LIB_FILES}
                      ${VID_TARGET_${_target_upper}_VMPCC_OUTPUTS}
                      ${VID_TARGET_${_target_upper}_VMPASM_SOURCES})

  # check if additional sources besides the lllib are defined, if not raise error
  if("${_vmpasm_sources}" STREQUAL "${VID_TARGET_${_target_upper}_VMP_LIB_FILES}")
    message(FATAL_ERROR "Target ${_target} has no sources defined. Use vid_add_sources_vmp() to add sources")
  endif()

  # append full path lowlevel library files to vmpasm sources list
  list(APPEND VID_TARGET_${_target_upper}_VMPASM_SOURCES ${_vmp_lib_files})
  # set list with vmpasm outputs
  set(VID_TARGET_${_target_upper}_VMPASM_OUTPUTS ${_vmpasm_out_files})
  # set list with vmpasm commands
  set(VID_TARGET_${_target_upper}_VMPASM_COMMANDS ${VMPASM_EXECUTABLE} ${_vmpasm_flags} ${_vmpasm_sources})

  # adapt path of vmpcc outputs, vmpasm out files and vmpasm mandatory out files,
  # if working directory is not current binary directory
  if(NOT ${_working_dir} STREQUAL ".")
    list(TRANSFORM VID_TARGET_${_target_upper}_VMPCC_OUTPUTS PREPEND ${_working_dir}/)
    list(TRANSFORM _vmpasm_out_files PREPEND ${_working_dir}/)
    list(TRANSFORM _vmpasm_mandatory_out_files PREPEND ${_working_dir}/)
  endif()

  # check if a target should be created
  if(NOT ${_ARG_NO_TARGET})
    # assemble sources with vmpasm
    add_custom_command(OUTPUT ${_vmpasm_out_files}
      COMMAND ${VMPASM_EXECUTABLE} ${_vmpasm_flags} ${_vmpasm_sources}
      DEPENDS ${VID_TARGET_${_target_upper}_VMPCC_OUTPUTS} ${VID_TARGET_${_target_upper}_VMPASM_SOURCES}
      WORKING_DIRECTORY ${_working_dir}
      VERBATIM COMMENT "Assembling ${_target} ...")
  endif()

  # define build target outputs
  set(_bin_file ${_target}.bin)
  set(_map_file ${_target}.map)

  # define list with map input files
  set(_map_input_files ${_vmpasm_memory_map_file})
  # check if niceout should be in map file
  if(${_ARG_NICEOUT})
    # append niceout to map file list
    list(APPEND _map_input_files ${_vmpasm_nice_memory_imem_file})
  else()
    # append hw out to map file list
    list(APPEND _map_input_files ${_vmpasm_hw_asm_file})
  endif()

  # set list with binary outputs
  set(VID_TARGET_${_target_upper}_BINARY_OUTPUTS ${_map_file} ${_bin_file})
  # set list with binary commands
  set(VID_TARGET_${_target_upper}_BINARY_COMMANDS cat ${_map_input_files} > ${_map_file} && cat ${_bin_input_files} > ${_bin_file})

  # check if a target should be created
  if(NOT ${_ARG_NO_TARGET})
    # concatenate vmpasm output files to map and bin file
    add_custom_command(OUTPUT ${_working_dir}/${_map_file} ${_working_dir}/${_bin_file}
      COMMAND cat ${_map_input_files} > ${_map_file}
      COMMAND cat ${_bin_input_files} > ${_bin_file}
      DEPENDS ${_vmpasm_mandatory_out_files}
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

    # set property for bin2h include directory of this target
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

  # adapt path of vmpasm outputs, binary outputs and bin2h outputs (vmpcc outputs are already adapted),
  # if working directory is not current binary directory
  if(NOT ${_working_dir} STREQUAL ".")
    list(TRANSFORM VID_TARGET_${_target_upper}_VMPASM_OUTPUTS PREPEND ${_working_dir}/)
    list(TRANSFORM VID_TARGET_${_target_upper}_BINARY_OUTPUTS PREPEND ${_working_dir}/)
    list(TRANSFORM VID_TARGET_${_target_upper}_BIN2H_OUTPUTS PREPEND ${_working_dir}/)
  endif()

  # set combined list with all external sources (outputs form commands that are sources to other commands are not included)
  set(VID_TARGET_${_target_upper}_SOURCES ${VID_TARGET_${_target_upper}_VMPCC_SOURCES} ${VID_TARGET_${_target_upper}_VMPASM_SOURCES})
  # set combined list with all generated outputs
  set(VID_TARGET_${_target_upper}_OUTPUTS ${VID_TARGET_${_target_upper}_VMPCC_OUTPUTS} ${VID_TARGET_${_target_upper}_VMPASM_OUTPUTS}
                                          ${VID_TARGET_${_target_upper}_BINARY_OUTPUTS} ${VID_TARGET_${_target_upper}_BIN2H_OUTPUTS})
  # set vmpcc commands to commands list
  set(VID_TARGET_${_target_upper}_COMMANDS ${VID_TARGET_${_target_upper}_VMPCC_COMMANDS})
  # get length of commands list
  list(LENGTH VID_TARGET_${_target_upper}_COMMANDS _cmd_length)
  # if list of commands contains something add && to list
  if(${_cmd_length} GREATER "0")
    list(APPEND VID_TARGET_${_target_upper}_COMMANDS &&)
  endif()
  # append vmpasm and binary commands to commands list
  list(APPEND VID_TARGET_${_target_upper}_COMMANDS ${VID_TARGET_${_target_upper}_VMPASM_COMMANDS} && ${VID_TARGET_${_target_upper}_BINARY_COMMANDS})
  # check if a header file with binary code should be created and append command to commands list
  if(${_ARG_BIN2H})
    list(APPEND VID_TARGET_${_target_upper}_COMMANDS && ${VID_TARGET_${_target_upper}_BIN2H_COMMANDS})
  endif()

  # add dependencies to link this target to others
  if(DEFINED _ARG_DEPENDENCIES AND (NOT ${_ARG_NO_TARGET}))
    add_dependencies(build_${_target} ${_ARG_DEPENDENCIES})
  endif()
endmacro(vid_build_target)

#[=======================================================================[.rst:
.. cmake:command:: vid_build_target_vmp

  Macro for building a videantis v-MP build target.

  The cmake module defines for every build target a define for assembler
  and compiler with the target cpu. E.g. v-MP 4.0 target cpu creates the
  define `__mp400__`.

  This is a wrapper macro for :cmake:command:`vid_build_target`.

  .. code-block:: cmake

    vid_build_target_vmp(target [DEPENDENCIES dependency1 [dependency2 ...]]
        [WORKING_DIRECTORY dir] [BIN2H] [NICEOUT] [NO_TARGET] [NO_ALL])

  Function parameters:

  ``target``
    name of an already defined build target
  ``DEPENDENCIES``
    add one or more cmake dependencies
  ``WORKING_DIRECTORY``
    define working directory (default: current cmake binary directory)
  ``BIN2H``
    create header file with binary code
  ``NICEOUT``
    create map file with --out-nice-memory-imem vspasm option output
  ``NO_TARGET``
    create no cmake targets
  ``NO_ALL``
    build target won't be added to ALL target of the build
#]=======================================================================]
macro(vid_build_target_vmp _target)
  cmake_parse_arguments(_ARG "BIN2H;NICEOUT;NO_TARGET;NO_ALL" "WORKING_DIRECTORY"
    "DEPENDENCIES" ${ARGN})

  # unset bin2h from possible prior calls of vid_build_target_vmp()
  unset(_bin2h)
  # check if BIN2H is set
  if(${_ARG_BIN2H})
    set(_bin2h BIN2H)
  endif()

  # unset niceout from possible prior calls of vid_build_target_vmp()
  unset(_niceout)
  # check if NICEOUT is set
  if(${_ARG_NICEOUT})
    set(_niceout NICEOUT)
  endif()

  # unset no_target from possible prior calls of vid_build_target_vmp()
  unset(_no_target)
  # check if NO_TARGET is set
  if(${_ARG_NO_TARGET})
    set(_no_target NO_TARGET)
  endif()

  # unset no_all from possible prior calls of vid_build_target_vmp()
  unset(_no_all)
  # check if NO_ALL is set
  if(${_ARG_NO_ALL})
    set(_no_all NO_ALL)
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_build_target_vmp() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  vid_build_target(${_target} WORKING_DIRECTORY ${_ARG_WORKING_DIRECTORY} DEPENDENCIES ${_ARG_DEPENDENCIES} ${_bin2h} ${_niceout} ${_no_target} ${_no_all})
endmacro(vid_build_target_vmp)
