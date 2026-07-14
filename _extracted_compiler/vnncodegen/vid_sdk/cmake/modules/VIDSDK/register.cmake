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
# FILENAME:    register.cmake
#
# DESCRIPTION: Support file for videantis SDK to register components
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

#[=======================================================================[.rst:
.. cmake:command:: vid_register_file

  Function for registering a file to the videantis build process.
  The registered file can contain a subdirectory within the name.

  .. code-block:: cmake

    vid_register_file(file identifier [SP])

  Function parameters:

  ``file``
    file for videantis build process
  ``identifier``
    identifier (library or project name) for the build process to create an unique variable
  ``SP``
    register files for v-SP (v-MP is default; option only available, if v-SP tool chain is available)

  Exposed global properties:

  ``VID_REGISTERED_FILES_VMPCC``
    list of registered <FILE_IDENTIFIER> for vmpcc cl files
  ``VID_REGISTERED_FILES_VMPASM``
    list of registered <FILE_IDENTIFIER> for vmpasm asm files
  ``VID_REGISTERED_FILES_VSPGCC``
    list of registered <FILE_IDENTIFIER> for vspgcc c files
  ``VID_REGISTERED_FILES_VSPASM``
    list of registered <FILE_IDENTIFIER> for vspasm asm files
  ``<FILE_IDENTIFIER_FULL>_PATH``
    full path to registered file
  ``<FILE_IDENTIFIER_FULL>_ID``
    <IDENTIFIER>

  Placeholder:

  ``<IDENTIFIER>``
    identifier in all upper case letters
  ``<FILE_IDENTIFIER>``
    combination of identifier and filename without extension in all upper case letters
  ``<FILE_IDENTIFIER_FULL>``
    combination of identifier, filename without extension and tool in all upper case letters
#]=======================================================================]
function(vid_register_file _file _identifier)
  # default no options are available
  unset(_options)
  # check if v-SP tool chain is available
  if(VIDSDK_HAS_VSP_TOOLCHAIN)
    # define SP as option
    set(_options SP)
  endif()
  cmake_parse_arguments(_ARG "${_options}" ""
    "" ${ARGN})

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_register_file() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

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

  # extract filename
  get_filename_component(_filename ${_file} NAME_WE)
  # extract file extension in upper case letters
  string(REGEX MATCH "\\.([a-zA-Z]+)$" _ ${_file})
  string(TOUPPER ${CMAKE_MATCH_1} _file_ext)

  if(_ARG_SP)
    # check for correct file extension
    if(NOT ((${_file_ext} IN_LIST VSPGCC_SUPPORTED_FILE_TYPES) OR (${_file_ext} IN_LIST VSPASM_SUPPORTED_FILE_TYPES)))
      message(FATAL_ERROR "Only ${VSPGCC_SUPPORTED_FILE_TYPES} and ${VSPASM_SUPPORTED_FILE_TYPES} files are supported")
    endif()

    # set default tool vspgcc
    set(_tool vspgcc)
    # check if file extension is supported by vspasm
    if(${_file_ext} IN_LIST VSPASM_SUPPORTED_FILE_TYPES)
      # set tool to vmpasm
      set(_tool vspasm)
    endif()
  else()
    # check for correct file extension
    if(NOT ((${_file_ext} IN_LIST VMPCC_SUPPORTED_FILE_TYPES) OR (${_file_ext} IN_LIST VMPASM_SUPPORTED_FILE_TYPES)))
      message(FATAL_ERROR "Only ${VMPCC_SUPPORTED_FILE_TYPES} and ${VMPASM_SUPPORTED_FILE_TYPES} files are supported")
    endif()

    # set default tool vmpcc
    set(_tool vmpcc)
    # check if file extension is supported by vmpasm
    if(${_file_ext} IN_LIST VMPASM_SUPPORTED_FILE_TYPES)
      # set tool to vmpasm
      set(_tool vmpasm)
    endif()
  endif()
  # create all upper case string of tool
  string(TOUPPER ${_tool} _tool_upper)

  # create file identifier
  string(TOUPPER ${_identifier}_${_filename} _file_identifier)
  # get registered files of actual type
  get_property(_vid_register_files_${_tool} GLOBAL PROPERTY VID_REGISTERED_FILES_${_tool_upper})
  # check if file identifier already exists
  if(${_file_identifier} IN_LIST _vid_register_files_${_tool})
    message(FATAL_ERROR "Filename in combination with the identifier is already registered. "
    "(Only the filename (not the full path) is used to create unique ID. "
    "If you have files with the same name use a different identifier.)")
  endif()
  # add file identifier to list of registered files
  set_property(GLOBAL APPEND PROPERTY VID_REGISTERED_FILES_${_tool_upper} ${_file_identifier})

  # create full file identifier
  set(_file_identifier_full ${_file_identifier}_${_tool_upper})
  # save full path to registered file
  set_property(GLOBAL PROPERTY ${_file_identifier_full}_PATH ${_file_path})
  # set identifier as upper case string
  string(TOUPPER ${_identifier} _identifier_upper)
  # save identifier for registered directory
  set_property(GLOBAL PROPERTY ${_file_identifier_full}_ID ${_identifier_upper})
  # build reverse index: identifier -> list of matching file identifiers
  # this allows _add_sources to do a direct O(k) lookup instead of scanning all registered files
  set_property(GLOBAL APPEND PROPERTY VID_REGISTERED_FILES_${_tool_upper}_ID_${_identifier_upper} ${_file_identifier})
endfunction(vid_register_file)

#[=======================================================================[.rst:
.. cmake:command:: vid_register_file_vmp

  Function for registering a v-MP file to the videantis build process.
  This is a wrapper function for :cmake:command:`vid_register_file` with no parameter set.

  .. code-block:: cmake

    vid_register_file_vmp(file identifier)

  Function parameters:

  ``file``
    file for videantis build process
  ``identifier``
    identifier (library or project name) for the build process to create a unique variable
#]=======================================================================]
function(vid_register_file_vmp _file _identifier)
  # call base function for registering a v-MP file
  vid_register_file(${_file} ${_identifier})
endfunction(vid_register_file_vmp)

#[=======================================================================[.rst:
.. cmake:command:: vid_register_include_dir

  Function for registering an include directory to the videantis build process.
  Directories are registered per default for vmpcc usage.

  .. code-block:: cmake

    vid_register_include_dir(dir identifier [VMPASM] [VSPGCC] [VSPASM])

  Function parameters:

  ``dir``
    dir for videantis build process
  ``identifier``
    identifier (library or project name) for the build process to create a unique variable
  ``VMPASM``
    flag to register a include directory for vmpasm
  ``VSPGCC``
    flag to register a include directory for vspgcc (option only available, if v-SP tool chain is available)
  ``VSPASM``
    flag to register a include directory for vspasm (option only available, if v-SP tool chain is available)

  Exposed global properties:

  ``VID_REGISTERED_INC_DIRS_VMPCC``
    list of registered <DIR_IDENTIFIER> for vmpcc
  ``VID_REGISTERED_INC_DIRS_VMPASM``
    list of registered <DIR_IDENTIFIER> for vmpasm
  ``VID_REGISTERED_INC_DIRS_VSPGCC``
    list of registered <DIR_IDENTIFIER> for vspgcc
  ``VID_REGISTERED_INC_DIRS_VSPASM``
    list of registered <DIR_IDENTIFIER> for vspasm
  ``<DIR_IDENTIFIER_FULL>_PATH``
    full path to registered include directory
  ``<DIR_IDENTIFIER_FULL>_ID``
    <IDENTIFIER>

  Placeholder:

  ``<IDENTIFIER>``
    identifier in all upper case letters
  ``<DIR_IDENTIFIER>``
    combination of identifier and directory name in all upper case letters
  ``<DIR_IDENTIFIER_FULL>``
    combination of identifier, directory name and tool name in all upper case letters
#]=======================================================================]
function(vid_register_include_dir _dir _identifier)
  # default only VMPASM option is available
  set(_options VMPASM)
  # check if v-SP tool chain is available
  if(VIDSDK_HAS_VSP_TOOLCHAIN)
    # define VSPGCC as option
    list(APPEND _options VSPGCC)
    # define VSPASM as option
    list(APPEND _options VSPASM)
  endif()
  cmake_parse_arguments(_ARG "${_options}"
    "" "" ${ARGN})

  # check if a dir with absolute path is given
  # otherwise set absolute path
  if(IS_ABSOLUTE ${_dir})
    set(_dir_path ${_dir})
    # remove path for directory
    # Note: good practice is to use this macro for directories in this folder or subfolders
    get_filename_component(_dir ${_dir_path} NAME)
  else()
    get_filename_component(_dir_path ${_dir} ABSOLUTE)
  endif()

  # check if specified directory exists
  if((NOT EXISTS ${_dir_path}) AND (IS_DIRECTORY ${_dir_path}))
    message(FATAL_ERROR "Directory ${_dir_path} does not exist")
  endif()

  # check if multiple tools are defined
  if((${_ARG_VMPASM} AND (${_ARG_VSPGCC} OR ${_ARG_VSPASM})) OR (${_ARG_VSPGCC} AND ${_ARG_VSPASM}))
    message(FATAL_ERROR "Only one tool can be defined at the function call of vid_register_include_dir()")
  endif()

  # raise a warning for the case some arguments cannot parsed
  if(DEFINED _ARG_UNPARSED_ARGUMENTS)
    message(WARNING "vid_register_include_dir() cannot parse the following arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()

  # set default tool vmpcc
  set(_tool "vmpcc")
  # if _VMPASM is set, set tool vmpasm
  if(${_ARG_VMPASM})
    set(_tool "vmpasm")
   # if _ARG_VSPGCC is set, set tool vspgcc
  elseif(${_ARG_VSPGCC})
    set(_tool "vspgcc")
  # if _ARG_VSPASM is set, set tool vspasm
  elseif(${_ARG_VSPASM})
    set(_tool "vspasm")
  endif()
  # set tool as string in upper case
  string(TOUPPER ${_tool} _tool_upper)

  # create dir identifier
  string(TOUPPER ${_identifier}_${_dir} _dir_identifier)
  string(REPLACE "." "DOT" _dir_identifier ${_dir_identifier})
  # get registered files of actual type
  get_property(_vid_register_inc_dirs_${_tool} GLOBAL PROPERTY VID_REGISTERED_INC_DIRS_${_tool_upper})
  # check if dir identifier already exists
  if(${_dir_identifier} IN_LIST _vid_register_inc_dirs_${_tool})
    message(FATAL_ERROR "Directory in combination with the identifier is already registered. "
      "(Only the directory (not the full path) is used to create unique ID. "
      "If you have directories with the same name use a different identifier.)")
  endif()
  # add file identifier to list of registered files
  set_property(GLOBAL APPEND PROPERTY VID_REGISTERED_INC_DIRS_${_tool_upper} ${_dir_identifier})

  # create full file identifier
  set(_dir_identifier_full ${_dir_identifier}_${_tool_upper})
  # save full path to registered directory
  set_property(GLOBAL PROPERTY ${_dir_identifier_full}_PATH ${_dir_path})
  # set identifier as upper case string
  string(TOUPPER ${_identifier} _identifier_upper)
  # save identifier for registered directory
  set_property(GLOBAL PROPERTY ${_dir_identifier_full}_ID ${_identifier_upper})
  # build reverse index: identifier -> list of matching dir identifiers
  # this allows _add_includes to do a direct O(k) lookup instead of scanning all registered dirs
  set_property(GLOBAL APPEND PROPERTY VID_REGISTERED_INC_DIRS_${_tool_upper}_ID_${_identifier_upper} ${_dir_identifier})
endfunction(vid_register_include_dir)

#[=======================================================================[.rst:
.. cmake:command:: vid_register_include_dir_vmpcc

  Function for registering an include directory for vmpcc to the videantis build process.
  This is a wrapper function for :cmake:command:`vid_register_include_dir` with no parameter set.

  .. code-block:: cmake

    vid_register_include_dir_vmpcc(dir identifier)

  Function parameters:

  ``dir``
    dir for videantis build process
  ``identifier``
    identifier (library or project name) for the build process to create a unique variable
#]=======================================================================]
function(vid_register_include_dir_vmpcc _dir _identifier)
  # call base function for registering a vmpcc include dir
  vid_register_include_dir(${_dir} ${_identifier})
endfunction(vid_register_include_dir_vmpcc)

#[=======================================================================[.rst:
.. cmake:command:: vid_register_include_dir_vmpasm

  Function for registering an include directory for vmpasm to the videantis build process.
  This is a wrapper function for :cmake:command:`vid_register_include_dir` with VMPASM parameter set.

  .. code-block:: cmake

    vid_register_include_dir_vmpasm(dir identifier)

  Function parameters:

  ``dir``
    dir for videantis build process
  ``identifier``
    identifier (library or project name) for the build process to create a unique variable
#]=======================================================================]
function(vid_register_include_dir_vmpasm _dir _identifier)
  # call base function for registering a vmpasm include dir
  vid_register_include_dir(${_dir} ${_identifier} VMPASM)
endfunction(vid_register_include_dir_vmpasm)
