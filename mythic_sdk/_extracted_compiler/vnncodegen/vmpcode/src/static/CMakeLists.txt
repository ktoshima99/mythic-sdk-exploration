# ++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++
#
# CONFIDENTIAL AND PROPRIETARY INFORMATION
# Copyright 2004 - 2026 videantis GmbH
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
# FILENAME:    CMakeLists.txt
#
# DESCRIPTION: Cmake file to add v-MP sources of vnncodegen/vmpcode/src/static
#
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++++

# register source files to identifier vid_vmp_static
vid_register_file_vmp(vid_vmp_boot_loader_simple.asm vid_vmp_static)
vid_register_file_vmp(defs.asm                       vid_vmp_static)
vid_register_file_vmp(status.asm                     vid_vmp_static)
#vid_register_file_vmp(msg_standalone.asm             vid_vmp_static)
vid_register_file_vmp(msg_handshake.asm              vid_vmp_static)
vid_register_file_vmp(msg.asm                        vid_vmp_static)
vid_register_file_vmp(main.asm                       vid_vmp_static)
vid_register_file_vmp(main_defs.asm                  vid_vmp_static)
vid_register_file_vmp(main_dsections.asm             vid_vmp_static)
vid_register_file_vmp(main_preload.asm               vid_vmp_static)
vid_register_file_vmp(dma.asm                        vid_vmp_static)

# register include dirs to identifier vid_vmp_static
vid_register_include_dir_vmpasm(. vid_vmp_static)
vid_register_include_dir_vmpcc(. vid_vmp_static)
