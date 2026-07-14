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
# DESCRIPTION: Support file for videantis SDK to register v-SP components
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

#[=======================================================================[.rst:
.. cmake:command:: vid_register_file_vsp

  Function for registering a v-SP file to the videantis build process.
  This is a wrapper function for :cmake:command:`vid_register_file` with SP parameter set.

  .. code-block:: cmake

    vid_register_file_vsp(file identifier)

  Function parameters:

  ``file``
    file for videantis build process
  ``identifier``
    identifier (library or project name) for the build process to create a unique variable
#]=======================================================================]
function(vid_register_file_vsp _file _identifier)
  # call base function for registering a v-SP file
  vid_register_file(${_file} ${_identifier} SP)
endfunction(vid_register_file_vsp)

#[=======================================================================[.rst:
.. cmake:command:: vid_register_include_dir_vspgcc

  Function for registering an include directory for vspgcc to the videantis build process.
  This is a wrapper function for :cmake:command:`vid_register_include_dir` with VSPGCC parameter set.

  .. code-block:: cmake

    vid_register_include_dir_vspgcc(dir identifier)

  Function parameters:

  ``dir``
    dir for videantis build process
  ``identifier``
    identifier (library or project name) for the build process to create a unique variable
#]=======================================================================]
function(vid_register_include_dir_vspgcc _dir _identifier)
  # call base function for registering a vspgcc include dir
  vid_register_include_dir(${_dir} ${_identifier} VSPGCC)
endfunction(vid_register_include_dir_vspgcc)

#[=======================================================================[.rst:
.. cmake:command:: vid_register_include_dir_vspasm

  Function for registering an include directory for vspasm to the videantis build process.
  This is a wrapper function for :cmake:command:`vid_register_include_dir` with VSPASM parameter set.

  .. code-block:: cmake

    vid_register_include_dir_vspasm(dir identifier)

  Function parameters:

  ``dir``
    dir for videantis build process
  ``identifier``
    identifier (library or project name) for the build process to create a unique variable
#]=======================================================================]
function(vid_register_include_dir_vspasm _dir _identifier)
  # call base function for registering a vspasm include dir
  vid_register_include_dir(${_dir} ${_identifier} VSPASM)
endfunction(vid_register_include_dir_vspasm)
