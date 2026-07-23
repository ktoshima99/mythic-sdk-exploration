# noqa: E501
#
# Copyright (C) 2025 Mythic Inc. All rights reserved. Contains trade secrets and confidential and
# proprietary information.
#
# Subject to patents and patents pending.
#
# This source code is licensed for use by licensee under the terms of a written license agreement
# between Mythic, Inc. and the licensee. Redistribution, disclosure, modification, the making of
# derivative works or any other use of this source code, in whole or in part, without the prior
# written consent of Mythic, Inc. is strictly prohibited, except as explicitly permitted in the
# license agreement.
#
# This source code is provided "as is" with all faults and without any warranties of any kind, and
# Mythic, Inc. disclaims all warranties, whether statutory, express, or implied, including without
# limitation, any warranties of non-infringement, merchantability, and fitness or sufficiency for
# a particular purpose. In no event will Mythic, Inc. be liable for any direct, indirect, special,
# exemplary, incidental, or consequential damages, losses, or liabilities, of any kind, including
# without limitation procurement of substitute goods or services; loss of use, data, profits, or
# goodwill; or business interruption; however caused and under any theory of liability, whether in
# contract, strict liability, or tort (including negligence or otherwise), arising any way out of
# the use of this software, regardless whether such damages, losses, or liabilities were
# foreseeable or even if Mythic, Inc. has been advised of the possibility of such damages, losses,
# or liabilities.
#
"""
Read in an hdf5 file and analyze the timestep activity to generate performance estimates.

usage: python perf_analysis.py --hdf5-path <hdf5_file_path>
  hdf5_file_path: the hdf5 file that contains the timestep activity data
  example: python perf_analysis.py --hdf5-path perf_trace_dump.h5

The hdf5 file is expected to have the following structure:
 group      /
 group      /ace_calcs
 dataset    /ace_calcs/ace_tile_N (N is the tile number)
 group      /simd_calcs
 dataset    /simd_calcs/ace_tile_N (N is the tile number)
 dataset    /simd_calcs/host_interface_tile
 group      /sram_accesses
 dataset    /sram_accesses/ace_tile_N (N is the tile number)
 dataset    /sram_accesses/host_interface_tile

The ACE calculation data is stored in the following format:
  /ace_calcs/ace_tile_N/trans_id
  /ace_calcs/ace_tile_N/timestep
  /ace_calcs/ace_tile_N/num_inputs
  /ace_calcs/ace_tile_N/num_outputs

The SIMD calculation data is stored in the following format:
  /simd_calcs/ace_tile_N/trans_id
  /simd_calcs/ace_tile_N/timestep
  /simd_calcs/ace_tile_N/num_input_bytes
  /simd_calcs/ace_tile_N/num_secondary_input_bytes
  /simd_calcs/ace_tile_N/num_output_bytes

The SRAM access data is stored in the following format:
  /sram_accesses/ace_tile_N/trans_id
  /sram_accesses/ace_tile_N/timestep
  /sram_accesses/ace_tile_N/address
  /sram_accesses/ace_tile_N/size (size of the access in bytes)
  /sram_accesses/ace_tile_N/estimated_final_size
   (the estimated final size of the control struct access in bytes)
  /sram_accesses/ace_tile_N/access_type (read=0 or write=1)
  /sram_accesses/ace_tile_N/initiator (string name of the initiator)
  /sram_accesses/ace_tile_N/hwu_id (hardware unit id of the initiator)
  /sram_accesses/ace_tile_N/is_rmw (read-modify-write)
  /sram_accesses/ace_tile_N/category
   (0=no category, 1=data buffer, 2=control flow, 3=management, 4=debug)
  /sram_accesses/ace_tile_N/control_flow_type (0=none, ...)
"""

import h5py
import argparse
from collections import defaultdict
from pathlib import Path
import logging
import textwrap
import sys
from contextlib import contextmanager
import json

try:
    from yaspin import yaspin
except ImportError:  # graceful fallback if yaspin isn't installed
    yaspin = None


# ----------------------------
# CLI / Logging Setup
# ----------------------------
def multiple_of_4(value: str) -> int:
    """Ensure value is a multiple of 4."""
    ivalue = int(value)
    if ivalue % 4 != 0:
        raise argparse.ArgumentTypeError(f"{value} is not a multiple of 4")
    return ivalue


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze the timestep activity from an HDF5 file."
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        dest="hdf5_path",
        required=True,
        help="File path to HDF5 containing timestep-based activity data",
    )
    parser.add_argument(
        "--num-aces",
        type=multiple_of_4,
        default=24,
        help="Number of ACEs per chip in integer multiples of 4 (default: %(default)s)"
    )
    parser.add_argument(
        "--digital-npu-log-path",
        type=Path,
        dest="npu_log_path",
        help="File path to JSON containing digital NPU performance data",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser.parse_args()


class SramCalculationMethodFilter(logging.Filter):
    """Logging filter to only allow messages for the specified SRAM calculation method."""

    def __init__(self, enabled_method):
        """Initialize filter with enabled method."""
        self.enabled_method = enabled_method

    def filter(self, record):
        """Filter log records based on SRAM calculation method."""
        method = getattr(record, "sram_method", None)
        if method is None:
            return True  # unrelated log messages pass through
        return method == self.enabled_method


def setup_logging(level_name: str):
    """Set up logging configuration."""
    log_level = getattr(logging, level_name.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(
            f"Invalid logging level: {level_name}. "
            "Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    handler = logging.StreamHandler()
    handler.addFilter(SramCalculationMethodFilter("accesses"))
    logging.basicConfig(
        level=log_level, format="%(message)s", handlers=[handler]
    )


# ----------------------------
# Progress Spinner
# ----------------------------
@contextmanager
def progress_spinner(text: str, enabled: bool):
    """
    Wrap yaspin into progress_spinner.

    Spinner is disabled if:
      - yaspin is not installed
      - logging is DEBUG (to avoid interleaving)
      - stderr is not a TTY
      - enabled flag is False
    """
    spinner_active = (
        enabled and yaspin is not None and sys.stderr.isatty()
        and logging.getLogger().level > logging.DEBUG
    )
    if spinner_active:
        with yaspin(text=text) as sp:
            try:
                yield sp
                sp.ok("✔")
            except Exception:
                sp.fail("✖")
                raise
    else:
        logging.debug(text + " ...")

        class _Noop:
            def __init__(self): pass
            def ok(self, *_): pass
            def fail(self, *_): pass
            @property
            def text(self): return ""
            @text.setter
            def text(self, _): pass
        try:
            yield _Noop()
        except Exception:
            raise


# ----------------------------
# Constants
# ----------------------------
CLOCK_PERIOD_NS = 1  # Digital clock period in nanoseconds - 1 GHz

# ACE constants
# --------------------------------------------------------------------------------------
ACE_DURATION_NS = 160  # Typical ACE calculation duration in nanoseconds
ACE_DURATION_SEC = ACE_DURATION_NS / 1e9  # Typical ACE calculation duration in seconds
ACE_TILE_X_DIM_MM = 8.24  # ACE tile X dimension in mm (pre-shrink)
ACE_TILE_Y_DIM_MM = 6.32  # ACE tile Y dimension in mm (pre-shrink)
ACE_TILE_X_DIM_PS_MM = ACE_TILE_X_DIM_MM * 0.9  # ACE tile X dimension in mm (post-shrink)
ACE_TILE_Y_DIM_PS_MM = ACE_TILE_Y_DIM_MM * 0.9  # ACE tile Y dimension in mm (post-shrink)
ACE_TILE_AREA_PS_MM2 = ACE_TILE_X_DIM_PS_MM * ACE_TILE_Y_DIM_PS_MM  # ACE tile die area mm^2

CYCLES_IN_A_TIMESTEP = ACE_DURATION_NS / CLOCK_PERIOD_NS

# SIMD constants
# --------------------------------------------------------------------------------------
SIMD_VECTOR_WIDTH = 8  # Number of vector lanes in SIMD Engine

# SRAM constants
# --------------------------------------------------------------------------------------
# SRAM constants for measuring activity based on number of Bytes
# --------------------------------------------------------------------------------------
IDEAL_SRAM_BYTES_READ_PER_CYCLE = 128
IDEAL_SRAM_BYTES_WRITTEN_PER_CYCLE = 128
SRAM_PERFORMANCE_SCALING_FACTOR = 0.5
SRAM_BYTES_READ_PER_CYCLE = IDEAL_SRAM_BYTES_READ_PER_CYCLE * SRAM_PERFORMANCE_SCALING_FACTOR
SRAM_BYTES_WRITTEN_PER_CYCLE = IDEAL_SRAM_BYTES_WRITTEN_PER_CYCLE * SRAM_PERFORMANCE_SCALING_FACTOR
SRAM_BYTES_READ_PER_ACE_OPERATION = SRAM_BYTES_READ_PER_CYCLE * ACE_DURATION_NS / CLOCK_PERIOD_NS
SRAM_BYTES_WRITTEN_PER_ACE_OPERATION = SRAM_BYTES_WRITTEN_PER_CYCLE * ACE_DURATION_NS / CLOCK_PERIOD_NS

# SRAM constants for measuring activity based on number of SRAM Accesses
# --------------------------------------------------------------------------------------
SRAM_ACCESS_PORTS = 8  # Actually 9, with one for GNOC & TOM not accounted for yet
SRAM_ACCESSES_PERFORMANCE_SCALING_FACTOR = 0.6  # Accounts for SRAM bank conflicts - a guess for now
TOTAL_SRAM_ACCESSES_PER_CYCLE = SRAM_ACCESS_PORTS * SRAM_ACCESSES_PERFORMANCE_SCALING_FACTOR
TOTAL_SRAM_ACCESSES_PER_TIMESTEP = TOTAL_SRAM_ACCESSES_PER_CYCLE * CYCLES_IN_A_TIMESTEP

# This is temporary and once we have stopped using older trace files soon, then we can just remove
# this and delete the related code. So I don't see a need to make this configurable.
USING_NUM_ACCESSES = True  # turn this to false when using an older trace file


# ----------------------------
# Performance Analysis
# ----------------------------
def analyze_file(args, spinner_enabled: bool):
    """Analyze the HDF5 file and print performance estimates."""
    # Get HDF5 file path from script arguments
    hdf5_path = args.hdf5_path
    # Get digital NPU JSON path from script arguments
    npu_log_path = args.npu_log_path

    npu_perf_data = defaultdict()
    if npu_log_path is not None:
        with progress_spinner(f"Opening digital NPU JSON file: {npu_log_path}", spinner_enabled):
            try:
                with open(npu_log_path, "r") as file:
                    contents = json.load(file)
                    npu_perf_data["fps"] = contents["profiling"]["fps"]
                    npu_perf_data["cycles"] = contents["profiling"]["total_cycles"]
                    npu_perf_data["mac_utilization"] = contents["profiling"]["mac_utilization_pct"]
                    npu_perf_data["macs_in_bn"] = contents["profiling"]["macs_bn"]
                    npu_perf_data["frequency"] = contents["system_config"]["sys"]["frequency"]
                    npu_perf_data["cores"] = contents["system_config"]["sys"]["nmps"]
            except FileNotFoundError:
                raise FileNotFoundError(f"Reading data from the digital NPU JSON file failed - {npu_log_path}")
            except KeyError:
                raise KeyError(f"Missing key when parsing the digital NPU JSON file - {npu_log_path}")
            except Exception as e:
                raise Exception(f"Error parsing data from digital NPU JSON file - {npu_log_path}") from e

    # Keep a top-level latency var for analog + digital sum
    frame_latency_ns = 0

    # Define constants for ACE Tile topology
    PER_CHIP_ACE_COUNT = args.num_aces  # No. of ACEs per chip (provided as script argument)
    PER_ACE_TILE_COUNT = 4  # No. of ACEs per ACE tile
    PER_CHIP_ACE_TILE_COUNT = PER_CHIP_ACE_COUNT // PER_ACE_TILE_COUNT  # No. of ACE tiles per chip

    with progress_spinner(f"Opening HDF5 file: {hdf5_path}", spinner_enabled):
        f = h5py.File(hdf5_path, "r")

    try:
        # Data structures
        control_bytes_per_tile = defaultdict(int)
        control_accesses_per_tile = defaultdict(int)
        control_bytes_read_per_tile = defaultdict(int)
        control_read_accesses_per_tile = defaultdict(int)
        control_bytes_written_per_tile = defaultdict(int)
        control_write_accesses_per_tile = defaultdict(int)
        data_bytes_per_tile = defaultdict(int)
        data_accesses_per_tile = defaultdict(int)
        data_bytes_read_per_tile = defaultdict(int)
        data_read_accesses_per_tile = defaultdict(int)
        data_bytes_written_per_tile = defaultdict(int)
        data_write_accesses_per_tile = defaultdict(int)
        other_bytes_per_tile = defaultdict(int)
        simd_bytes_per_tile = defaultdict(int)

        # Timestamp accesses
        ts_accesses = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "read_data_bytes": 0,
                    "read_control_bytes": 0,
                    "write_data_bytes": 0,
                    "write_control_bytes": 0,
                    "read_data_accesses": 0,
                    "read_control_accesses": 0,
                    "write_data_accesses": 0,
                    "write_control_accesses": 0,
                    "all_accesses": 0,
                }
            )
        )

        # ACE counts
        ace_counts = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "ace_operations": 0,
                    "ace_mac_count": 0,
                }
            )
        )

        # SIMD counts
        simd_counts = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "simd_operations": 0,
                    "simd_input_bytes": 0,
                }
            )
        )

        # Parse SRAM accesses
        sram_group = f.get("/sram_accesses")
        if sram_group is None:
            logging.warning("Group '/sram_accesses' not found in HDF5; skipping SRAM parsing.")
            sram_tiles = []
        else:
            sram_tiles = list(sram_group.keys())

        with progress_spinner(
            f"Parsing /sram_accesses ({len(sram_tiles)} tiles)",
            spinner_enabled,
        ) as sp:
            for idx, tile_name in enumerate(sram_tiles, start=1):
                if sp:  # Update message lightly to show progress
                    sp.text = f"Parsing /sram_accesses [{idx}/{len(sram_tiles)}] {tile_name}"
                tile_group = f[f"/sram_accesses/{tile_name}"]

                access_type = tile_group["access_type"][:]
                sizes = tile_group["size"][:]
                if USING_NUM_ACCESSES:
                    num_accesses = tile_group["num_accesses"][:]
                else:
                    num_accesses = [1] * len(sizes)
                category = tile_group["category"][:]
                timesteps = tile_group["timestep"][:]

                for i in range(len(sizes)):
                    timestep = timesteps[i]
                    ts_accesses[tile_name][timestep]["all_accesses"] += num_accesses[i]
                    if access_type[i] == 0:  # Read
                        if category[i] == 2:  # Control Flow
                            ts_accesses[tile_name][timestep]["read_control_bytes"] += sizes[i]
                            ts_accesses[tile_name][timestep]["read_control_accesses"] += num_accesses[i]
                        elif category[i] == 1:  # Data Buffer
                            ts_accesses[tile_name][timestep]["read_data_bytes"] += sizes[i]
                            ts_accesses[tile_name][timestep]["read_data_accesses"] += num_accesses[i]
                    elif access_type[i] == 1:  # Write
                        if category[i] == 2:  # Control Flow
                            ts_accesses[tile_name][timestep]["write_control_bytes"] += sizes[i]
                            ts_accesses[tile_name][timestep]["write_control_accesses"] += num_accesses[i]
                        elif category[i] == 1:  # Data Buffer
                            ts_accesses[tile_name][timestep]["write_data_bytes"] += sizes[i]
                            ts_accesses[tile_name][timestep]["write_data_accesses"] += num_accesses[i]

                    # Per-tile totals
                    if category[i] == 2:  # Control Flow
                        control_bytes_per_tile[tile_name] += sizes[i]
                        control_accesses_per_tile[tile_name] += num_accesses[i]
                        if access_type[i] == 0:
                            control_bytes_read_per_tile[tile_name] += sizes[i]
                            control_read_accesses_per_tile[tile_name] += num_accesses[i]
                        elif access_type[i] == 1:
                            control_bytes_written_per_tile[tile_name] += sizes[i]
                            control_write_accesses_per_tile[tile_name] += num_accesses[i]
                    elif category[i] == 1:  # Data Buffer
                        data_bytes_per_tile[tile_name] += sizes[i]
                        data_accesses_per_tile[tile_name] += num_accesses[i]
                        if access_type[i] == 0:
                            data_bytes_read_per_tile[tile_name] += sizes[i]
                            data_read_accesses_per_tile[tile_name] += num_accesses[i]
                        elif access_type[i] == 1:
                            data_bytes_written_per_tile[tile_name] += sizes[i]
                            data_write_accesses_per_tile[tile_name] += num_accesses[i]
                    else:
                        other_bytes_per_tile[tile_name] += sizes[i]

        # Parse ACE calcs
        ace_group = f.get("/ace_calcs")
        if ace_group is None:
            logging.warning(
                "Group '/ace_calcs' not found in HDF5; skipping ACE parsing."
            )
            ace_names = []
        else:
            ace_names = list(ace_group.keys())

        with progress_spinner(
            f"Parsing /ace_calcs ({len(ace_names)} ACEs)", spinner_enabled
        ) as sp:
            for idx, ace_name in enumerate(ace_names, start=1):
                if sp:
                    sp.text = f"Parsing /ace_calcs [{idx}/{len(ace_names)}] {ace_name}"
                ace_g = f[f"/ace_calcs/{ace_name}"]
                num_inputs = ace_g["num_inputs"][:]
                num_outputs = ace_g["num_outputs"][:]

                for ni, no in zip(num_inputs, num_outputs):
                    bucket = ace_counts[ace_name]["all"]
                    bucket["ace_operations"] += 1
                    bucket["ace_mac_count"] += int(ni) * int(no)

        # Parse SIMD calcs
        simd_group = f.get("/simd_calcs")
        if simd_group is None:
            logging.warning("Group '/simd_calcs' not found in HDF5; skipping SIMD parsing.")
            simd_names = []
        else:
            simd_names = list(simd_group.keys())

        with progress_spinner(f"Parsing /simd_calcs ({len(simd_names)} SIMDs)", spinner_enabled) as sp:
            for idx, simd_name in enumerate(simd_names, start=1):
                if sp:
                    sp.text = f"Parsing /simd_calcs [{idx}/{len(simd_names)}] {simd_name}"
                simd_g = f[f"/simd_calcs/{simd_name}"]
                timesteps_arr = simd_g["timestep"][:]

                for i in range(len(timesteps_arr)):
                    timestep = timesteps_arr[i]
                    simd_counts[simd_name][timestep]["simd_operations"] += 1
                    simd_counts[simd_name][timestep]["simd_input_bytes"] += simd_g["num_input_bytes"][i]

        # Compute stats & timings
        with progress_spinner(
            "Computing aggregate statistics", spinner_enabled
        ):
            tiles = sorted(ts_accesses.keys())

            logging.debug("SRAM Data Bytes Read per Tile:")
            for tile in tiles:
                logging.debug(f"{tile}: {data_bytes_read_per_tile[tile]:,}")
            logging.debug("SRAM Data Bytes Written per Tile:")
            for tile in tiles:
                logging.debug(f"{tile}: {data_bytes_written_per_tile[tile]:,}")
            logging.debug("SRAM Control Bytes Read per Tile:")
            for tile in tiles:
                logging.debug(f"{tile}: {control_bytes_read_per_tile[tile]:,}")
            logging.debug("SRAM Control Bytes Written per Tile:")
            for tile in tiles:
                logging.debug(f"{tile}: {control_bytes_written_per_tile[tile]:,}")
            logging.debug("SRAM Data Read Accesses per Tile:")
            for tile in tiles:
                logging.debug(f"{tile}: {data_read_accesses_per_tile[tile]:,}")
            logging.debug("SRAM Data Write Accesses per Tile:")
            for tile in tiles:
                logging.debug(f"{tile}: {data_write_accesses_per_tile[tile]:,}")
            logging.debug("SRAM Control Read Accesses per Tile:")
            for tile in tiles:
                logging.debug(f"{tile}: {control_read_accesses_per_tile[tile]:,}")
            logging.debug("SRAM Control Write Accesses per Tile:")
            for tile in tiles:
                logging.debug(f"{tile}: {control_write_accesses_per_tile[tile]:,}")

            # Determine timesteps
            timesteps = set()
            for tile in ts_accesses.keys():
                for timestep in ts_accesses[tile].keys():
                    timesteps.add(timestep)
            timesteps = sorted(timesteps)

        with progress_spinner(f"Estimating per-timestep durations ({len(timesteps)} timesteps)",
                              spinner_enabled,) as sp:
            # These "duration_ns" variables are based on taking a max over all tiles (not added
            # over all tiles)
            total_duration_ns = 0
            total_excess_simd_duration_ns = 0
            total_excess_sram_read_duration_ns = 0
            total_excess_sram_write_duration_ns = 0
            total_duration_accesses_ns = 0
            total_duration_accesses_no_simd_ns = 0
            total_excess_simd_duration_accesses_ns = 0
            total_excess_sram_duration_accesses_ns = 0

            total_simd_usage_ns = 0  # Added up over all tiles and timesteps

            for i, timestep in enumerate(timesteps, start=1):
                if sp:
                    sp.text = f"Estimating durations [{i}/{len(timesteps)}] (timestep={timestep})"

                logging.debug(f"Timestep {timestep}:")

                # Max across tiles for this timestep
                max_read_data_bytes_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["read_data_bytes"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                max_write_data_bytes_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["write_data_bytes"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                max_read_control_bytes_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["read_control_bytes"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                max_write_control_bytes_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["write_control_bytes"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                max_read_data_accesses_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["read_data_accesses"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                max_write_data_accesses_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["write_data_accesses"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                max_read_control_accesses_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["read_control_accesses"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                max_write_control_accesses_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["write_control_accesses"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                max_accesses_by_any_tile_this_timestep = max(
                    ts_accesses[tile][timestep]["all_accesses"]
                    for tile in ts_accesses
                    if timestep in ts_accesses[tile]
                ) if ts_accesses else 0

                # Log max values for this timestep
                logging.debug(
                    f"Max SRAM Data Bytes Read by Any Tile: "
                    f"{max_read_data_bytes_by_any_tile_this_timestep:,}"
                )
                logging.debug(
                    f"Max SRAM Data Bytes Written by Any Tile: "
                    f"{max_write_data_bytes_by_any_tile_this_timestep:,}"
                )
                logging.debug(
                    f"Max SRAM Control Bytes Read by Any Tile: "
                    f"{max_read_control_bytes_by_any_tile_this_timestep:,}"
                )
                logging.debug(
                    f"Max SRAM Control Bytes Written by Any Tile: "
                    f"{max_write_control_bytes_by_any_tile_this_timestep:,}"
                )
                logging.debug(
                    f"Max SRAM Data Read Accesses by Any Tile: "
                    f"{max_read_data_accesses_by_any_tile_this_timestep:,}"
                )
                logging.debug(
                    f"Max SRAM Data Write Accesses by Any Tile: "
                    f"{max_write_data_accesses_by_any_tile_this_timestep:,}"
                )
                logging.debug(
                    f"Max SRAM Control Read Accesses by Any Tile: "
                    f"{max_read_control_accesses_by_any_tile_this_timestep:,}"
                )
                logging.debug(
                    f"Max SRAM Control Write Accesses by Any Tile: "
                    f"{max_write_control_accesses_by_any_tile_this_timestep:,}"
                )
                logging.debug(
                    f"Max Total SRAM Accesses by Any Tile: "
                    f"{max_accesses_by_any_tile_this_timestep:,}"
                )

                # ptoth - this probably isn't right... we should total the values at
                # each tile first, then get the max.
                max_total_read_bytes_by_any_tile_this_timestep = (
                    max_read_data_bytes_by_any_tile_this_timestep
                    + max_read_control_bytes_by_any_tile_this_timestep
                )
                max_total_write_bytes_by_any_tile_this_timestep = (
                    max_write_data_bytes_by_any_tile_this_timestep
                    + max_write_control_bytes_by_any_tile_this_timestep
                )
                sram_read_estimated_duration_ns = (
                    max_total_read_bytes_by_any_tile_this_timestep / SRAM_BYTES_READ_PER_CYCLE
                ) * CLOCK_PERIOD_NS if SRAM_BYTES_READ_PER_CYCLE > 0 else 0

                # Find high SRAM read activity using the bytes-based estimate
                if (max_total_read_bytes_by_any_tile_this_timestep > SRAM_BYTES_READ_PER_ACE_OPERATION):
                    excess_read_bytes = (max_total_read_bytes_by_any_tile_this_timestep -
                                         SRAM_BYTES_READ_PER_ACE_OPERATION)
                    excess_duration_ns = ((excess_read_bytes / SRAM_BYTES_READ_PER_CYCLE) *
                                          CLOCK_PERIOD_NS) if SRAM_BYTES_READ_PER_CYCLE > 0 else 0
                    logging.debug(
                        f"[{timestep}] High SRAM Read Activity: "
                        f"{max_total_read_bytes_by_any_tile_this_timestep:,} bytes, "
                        f"Estimated Duration Impact: {excess_duration_ns:,.2f} ns",
                        extra={"sram_method": "bytes"},
                    )

                sram_write_estimated_duration_ns = (
                    (max_total_write_bytes_by_any_tile_this_timestep / SRAM_BYTES_WRITTEN_PER_CYCLE)
                    * CLOCK_PERIOD_NS if SRAM_BYTES_WRITTEN_PER_CYCLE > 0 else 0
                )

                # Find high SRAM write activity using the bytes-based estimate
                if (max_total_write_bytes_by_any_tile_this_timestep > SRAM_BYTES_WRITTEN_PER_ACE_OPERATION):
                    excess_write_bytes = (max_total_write_bytes_by_any_tile_this_timestep
                                          - SRAM_BYTES_WRITTEN_PER_ACE_OPERATION)
                    excess_duration_ns = (
                        (excess_write_bytes / SRAM_BYTES_WRITTEN_PER_CYCLE)
                        * CLOCK_PERIOD_NS if SRAM_BYTES_WRITTEN_PER_CYCLE > 0 else 0
                    )
                    logging.debug(
                        f"[{timestep}] High SRAM Write Activity: "
                        f"{max_total_write_bytes_by_any_tile_this_timestep:,} bytes, "
                        f"Estimated Duration Impact: {excess_duration_ns:,.2f} ns",
                        extra={"sram_method": "bytes"},
                    )

                # Estimate based on _all_ SRAM access types
                # The RTL design allows up to SRAM_ACCESS_PORTS accesses to proceed in parallel regardless of
                # whether they are reads or writes, so we can use the total accesses for a more accurate estimate
                # than splitting into reads and writes.
                sram_estimated_duration_accesses_ns = (
                    (max_accesses_by_any_tile_this_timestep / TOTAL_SRAM_ACCESSES_PER_CYCLE) * CLOCK_PERIOD_NS
                    if TOTAL_SRAM_ACCESSES_PER_CYCLE > 0 else 0
                )

                # Total SRAM activity using accesses-based estimate
                if (max_accesses_by_any_tile_this_timestep > TOTAL_SRAM_ACCESSES_PER_TIMESTEP):
                    excess_total_accesses = (
                        max_accesses_by_any_tile_this_timestep - TOTAL_SRAM_ACCESSES_PER_TIMESTEP
                    )
                    excess_duration_accesses_ns = (
                        (excess_total_accesses / TOTAL_SRAM_ACCESSES_PER_CYCLE) * CLOCK_PERIOD_NS
                        if TOTAL_SRAM_ACCESSES_PER_CYCLE > 0 else 0
                    )
                    logging.debug(
                        f"[{timestep}] High Total SRAM Activity: "
                        f"{max_accesses_by_any_tile_this_timestep:,} accesses, "
                        f"Estimated Duration Impact: {excess_duration_accesses_ns:,.2f} ns",
                        extra={"sram_method": "accesses"},
                    )

                # SIMD estimate
                longest_simd_estimated_duration_ns = 0
                for simd_name in simd_counts.keys():
                    simd_operations = 0
                    simd_estimated_duration_ns = 0
                    if timestep in simd_counts[simd_name]:
                        simd_operations = simd_counts[simd_name][timestep]["simd_operations"]
                        simd_input_bytes = simd_counts[simd_name][timestep]["simd_input_bytes"]
                        simd_bytes_per_tile[simd_name] += simd_input_bytes
                        simd_estimated_duration_ns = (
                            simd_input_bytes / SIMD_VECTOR_WIDTH * CLOCK_PERIOD_NS
                        )
                        total_simd_usage_ns += simd_estimated_duration_ns
                    longest_simd_estimated_duration_ns = max(
                        longest_simd_estimated_duration_ns,
                        simd_estimated_duration_ns,
                    )
                    if simd_operations > 0:
                        logging.debug(
                            f"[{timestep}] SIMD Operations on SIMD {simd_name}: {simd_operations}, "
                            f"Estimated Duration: {simd_estimated_duration_ns:,.2f} ns"
                        )

                timestep_duration_ns = max(
                    ACE_DURATION_NS,
                    sram_read_estimated_duration_ns,
                    sram_write_estimated_duration_ns,
                    longest_simd_estimated_duration_ns,
                )
                logging.debug(
                    f"[{timestep}] Estimated Duration: {timestep_duration_ns:,.2f} ns",
                    extra={"sram_method": "bytes"},
                )
                total_duration_ns += timestep_duration_ns

                if timestep_duration_ns == longest_simd_estimated_duration_ns:
                    total_excess_simd_duration_ns += (timestep_duration_ns - ACE_DURATION_NS)
                elif timestep_duration_ns == sram_read_estimated_duration_ns:
                    total_excess_sram_read_duration_ns += (timestep_duration_ns - ACE_DURATION_NS)
                elif timestep_duration_ns == sram_write_estimated_duration_ns:
                    total_excess_sram_write_duration_ns += (timestep_duration_ns - ACE_DURATION_NS)

                timestep_duration_accesses_ns = max(
                    ACE_DURATION_NS,
                    sram_estimated_duration_accesses_ns,
                    longest_simd_estimated_duration_ns,
                )
                logging.debug(
                    f"[{timestep}] Estimated Duration: {timestep_duration_accesses_ns:,.2f} ns",
                    extra={"sram_method": "accesses"},
                )
                total_duration_accesses_ns += timestep_duration_accesses_ns

                # see what the result would be if we ignored SIMD
                timestep_duration_accesses_no_simd_ns = max(
                    ACE_DURATION_NS,
                    sram_estimated_duration_accesses_ns,
                )
                logging.debug(
                    f"[{timestep}] Estimated Duration (ignoring SIMD): "
                    f"{timestep_duration_accesses_no_simd_ns:,.2f} ns",
                    extra={"sram_method": "accesses"},
                )
                total_duration_accesses_no_simd_ns += (
                    timestep_duration_accesses_no_simd_ns
                )

                # if the duration is less than or equal to the ACE duration, then do nothing
                if timestep_duration_accesses_ns <= ACE_DURATION_NS:
                    pass
                elif (timestep_duration_accesses_ns == longest_simd_estimated_duration_ns):
                    total_excess_simd_duration_accesses_ns += (timestep_duration_accesses_ns - ACE_DURATION_NS)
                elif (timestep_duration_accesses_ns == sram_estimated_duration_accesses_ns):
                    total_excess_sram_duration_accesses_ns += (timestep_duration_accesses_ns - ACE_DURATION_NS)

        # ---------------------------------------------------------------------------
        # Compute bottleneck duration as:
        #
        #    max(
        #        ACE critical-path latency,
        #        max SRAM read time across tiles,
        #        max SRAM write time across tiles,
        #        max SIMD time across tiles
        #    )
        #
        # The final estimated processing time is this bottleneck duration.
        # ---------------------------------------------------------------------------

        # Critical path ACE latency
        total_ace_duration_ns = len(timesteps) * ACE_DURATION_NS

        # Total across chip - Bytes
        # -------------------------------------------------------------------
        # Calculate the total time spent doing SRAM reads and total time
        # spent doing SRAM writes based on the total bytes read and written
        # across the entire chip.
        # This is primarily informational and doesn't tell us much about
        # system latency.
        total_data_bytes_read = sum(
            (int(v) for v in data_bytes_read_per_tile.values()), 0
        )
        total_data_bytes_written = sum(
            (int(v) for v in data_bytes_written_per_tile.values()), 0
        )
        total_control_bytes_read = sum(
            (int(v) for v in control_bytes_read_per_tile.values()), 0
        )
        total_control_bytes_written = sum(
            (int(v) for v in control_bytes_written_per_tile.values()), 0
        )
        total_bytes_read = total_data_bytes_read + total_control_bytes_read
        total_bytes_written = total_data_bytes_written + total_control_bytes_written

        # Total across chip - Accesses
        # -------------------------------------------------------------------
        # Calculate the total time spent doing SRAM reads and total time
        # spent doing SRAM writes based on the total number of SRAM accesses
        # across the entire chip.
        # This is primarily informational and doesn't tell us much about
        # system latency.
        total_data_read_accesses = sum(
            (int(v) for v in data_read_accesses_per_tile.values()), 0
        )
        total_data_write_accesses = sum(
            (int(v) for v in data_write_accesses_per_tile.values()), 0
        )
        total_control_read_accesses = sum(
            (int(v) for v in control_read_accesses_per_tile.values()), 0
        )
        total_control_write_accesses = sum(
            (int(v) for v in control_write_accesses_per_tile.values()), 0
        )
        total_read_accesses = total_data_read_accesses + total_control_read_accesses
        total_write_accesses = total_data_write_accesses + total_control_write_accesses
        total_sram_accesses = total_read_accesses + total_write_accesses

        # Duration using Bytes
        total_sram_read_duration_ns = (
            (total_bytes_read / SRAM_BYTES_READ_PER_CYCLE) * CLOCK_PERIOD_NS
            if SRAM_BYTES_READ_PER_CYCLE > 0 else 0
        )
        total_sram_write_duration_ns = (
            (total_bytes_written / SRAM_BYTES_WRITTEN_PER_CYCLE) * CLOCK_PERIOD_NS
            if SRAM_BYTES_WRITTEN_PER_CYCLE > 0 else 0
        )

        # Duration using Accesses
        total_sram_access_duration_ns = (
            (total_sram_accesses / TOTAL_SRAM_ACCESSES_PER_CYCLE) * CLOCK_PERIOD_NS
            if TOTAL_SRAM_ACCESSES_PER_CYCLE > 0 else 0
        )

        # Max across tiles - Bytes
        # -------------------------------------------------------------------
        # Find out which tile would take the longest time based on SRAM
        # bytes read and written by taking all of its bytes read across
        # all timesteps and then dividing by how many bytes we can read
        # per cycle. Then do the same thing for bytes written.
        # This is a best-case assuming SRAM bytes read and written can be
        # perfectly packed over time - other than the scaling factor used
        # above to account for SRAM bank collisions.
        tiles = sorted(ts_accesses.keys())
        max_bytes_read_by_any_tile = max(
            data_bytes_read_per_tile[tile] + control_bytes_read_per_tile[tile]
            for tile in tiles
        )
        max_bytes_written_by_any_tile = max(
            data_bytes_written_per_tile[tile] + control_bytes_written_per_tile[tile]
            for tile in tiles
        )
        max_sram_read_duration_ns = (
            (max_bytes_read_by_any_tile / SRAM_BYTES_READ_PER_CYCLE) * CLOCK_PERIOD_NS
            if SRAM_BYTES_READ_PER_CYCLE > 0 else 0
        )
        max_sram_write_duration_ns = (
            (max_bytes_written_by_any_tile / SRAM_BYTES_WRITTEN_PER_CYCLE) * CLOCK_PERIOD_NS
            if SRAM_BYTES_WRITTEN_PER_CYCLE > 0 else 0
        )

        # Max across tiles - Accesses
        # -------------------------------------------------------------------
        # Find out which tile would take the longest time based on SRAM
        # accesses by taking all of its accesses across all timesteps
        # and then dividing by how many accessess we can do per cycle.
        # This is a best-case assuming SRAM accesses can be perfectly
        # packed over time - other than the scaling factor used above to
        # account for SRAM bank collisions.
        max_accesses_by_any_tile = max(
            data_accesses_per_tile[tile] + control_accesses_per_tile[tile]
            for tile in tiles
        )
        max_sram_duration_accesses_ns = (
            (max_accesses_by_any_tile / TOTAL_SRAM_ACCESSES_PER_CYCLE) * CLOCK_PERIOD_NS
            if TOTAL_SRAM_ACCESSES_PER_CYCLE > 0 else 0
        )

        # total_simd_usage_ns already computed above when iterating over timesteps
        max_simd_usage_by_any_tile = (
            max(simd_bytes_per_tile.values(), default=0) / SIMD_VECTOR_WIDTH * CLOCK_PERIOD_NS
        )

        maximum_bottleneck_ns = max(
            total_ace_duration_ns,
            max_sram_read_duration_ns,
            max_sram_write_duration_ns,
            max_simd_usage_by_any_tile,
        )

        # bottleneck calculation using accesses
        maximum_bottleneck_accesses_ns = max(
            total_ace_duration_ns,
            max_sram_duration_accesses_ns,
            max_simd_usage_by_any_tile,
        )

        # Final reporting

        logging.debug(f"\nNumber of Timesteps: {len(timesteps):,}")

        if total_duration_ns < 1_000_000:
            time_unit_str = "us"
            conversion_factor = 1 / 1000
        else:
            time_unit_str = "ms"
            conversion_factor = 1 / 1_000_000

        def format_time(value_ns: float) -> str:
            """Return time string e.g. 123.45 ms."""
            return f"{value_ns * conversion_factor:,.2f} {time_unit_str}"

        def format_percent(part: float, total: float) -> str:
            """Return percent string e.g. 12.34% or N/A if total is zero."""
            if total == 0:
                return "N/A%"
            return f"{(part / total) * 100:.2f}%"

        logging.info(
            f"Analog NPU Total Estimated Processing Time for Compiled ONNX File: "
            f"{format_time(maximum_bottleneck_ns)}",
            extra={"sram_method": "bytes"},
        )

        if maximum_bottleneck_ns > 0:
            frame_latency_ns = maximum_bottleneck_ns
            fps = 1e9 / maximum_bottleneck_ns
            logging.info(
                f"Analog NPU Estimated Frame Rate for Compiled ONNX File: "
                f"{fps:,.2f} fps (frames per second)",
                extra={"sram_method": "bytes"},
            )

        logging.info(
            f"Analog NPU Total Estimated Processing Time for Compiled ONNX File: "
            f"{format_time(maximum_bottleneck_accesses_ns)}",
            extra={"sram_method": "accesses"},
        )

        if maximum_bottleneck_accesses_ns > 0:
            frame_latency_ns = maximum_bottleneck_accesses_ns
            fps = 1e9 / maximum_bottleneck_accesses_ns
            logging.info(
                f"Analog NPU Estimated Frame Rate for Compiled ONNX File: "
                f"{fps:,.2f} fps (frames per second)",
                extra={"sram_method": "accesses"},
            )

        logging.info(
            f"Critical Path ACE Latency: {format_time(total_ace_duration_ns)}"
        )

        num_ace_tiles = len(ace_names)

        logging.info(
            f"Maximum SRAM Read Time (over {num_ace_tiles} ACE tiles): "
            f"{format_time(max_sram_read_duration_ns)}",
            extra={"sram_method": "bytes"},
        )
        logging.info(
            f"Maximum SRAM Write Time (over {num_ace_tiles} ACE tiles): "
            f"{format_time(max_sram_write_duration_ns)}",
            extra={"sram_method": "bytes"},
        )
        logging.info(
            f"Maximum SRAM Read/Write Time (over {num_ace_tiles} ACE tiles): "
            f"{format_time(max_sram_duration_accesses_ns)}",
            extra={"sram_method": "accesses"},
        )
        logging.info(
            f"Maximum SIMD Operation Time (over {num_ace_tiles} ACE tiles): "
            f"{format_time(max_simd_usage_by_any_tile)}"
        )

        if num_ace_tiles > 0:
            logging.info(
                f"Average SRAM Read Time (averaged over {num_ace_tiles} ACE tiles): "
                f"{format_time(total_sram_read_duration_ns / num_ace_tiles)}",
                extra={"sram_method": "bytes"},
            )
            logging.info(
                f"Average SRAM Write Time (averaged over {num_ace_tiles} ACE tiles): "
                f"{format_time(total_sram_write_duration_ns / num_ace_tiles)}",
                extra={"sram_method": "bytes"},
            )
            logging.info(
                f"Average SRAM Read/Write Time (averaged over {num_ace_tiles} ACE tiles): "
                f"{format_time(total_sram_access_duration_ns / num_ace_tiles)}",
                extra={"sram_method": "accesses"},
            )

            logging.info(
                f"Average SIMD Operation Time (averaged over {num_ace_tiles} ACE tiles): "
                f"{format_time(total_simd_usage_ns / num_ace_tiles)}"
            )

        # --------------------------------------------------------------------
        # epoch based stats
        # --------------------------------------------------------------------
        # Total Time using SRAM Bytes
        logging.debug(
            f"Total Estimated Analog NPU Processing Time for Compiled ONNX File (epoch based): "
            f"{format_time(total_duration_ns)}",
            extra={"sram_method": "bytes"},
        )
        # Total Time using SRAM Accesses
        logging.debug(
            f"Total Estimated Analog NPU Processing Time for Compiled ONNX File (epoch based): "
            f"{format_time(total_duration_accesses_ns)}",
            extra={"sram_method": "accesses"},
        )
        logging.debug(
            f"Total Estimated Analog NPU Processing Time for Compiled ONNX File (epoch based) - "
            f"ignoring SIMD: "
            f"{format_time(total_duration_accesses_no_simd_ns)}",
            extra={"sram_method": "accesses"},
        )

        # Total ACE Time - minimum possible time if all else is ideal
        logging.debug(
            f"  Total ACE Latency: {format_time(total_ace_duration_ns)} "
            f"({format_percent(total_ace_duration_ns, total_duration_ns)} "
            f"of (epoch based) total latency)",
            extra={"sram_method": "bytes"},
        )
        logging.debug(
            f"  Total ACE Latency: {format_time(total_ace_duration_ns)} "
            f"({format_percent(total_ace_duration_ns, total_duration_accesses_ns)} "
            f"of (epoch based) total latency)",
            extra={"sram_method": "accesses"},
        )

        # Total Excess SRAM Read & Write Time using SRAM Bytes
        logging.debug(
            f"  Total Excess SRAM Read Latency: "
            f"{format_time(total_excess_sram_read_duration_ns)} "
            f"({format_percent(total_excess_sram_read_duration_ns, total_duration_ns)} "
            f"of (epoch based) total latency)",
            extra={"sram_method": "bytes"},
        )
        logging.debug(
            f"  Total Excess SRAM Write Latency: "
            f"{format_time(total_excess_sram_write_duration_ns)} "
            f"({format_percent(total_excess_sram_write_duration_ns, total_duration_ns)} "
            f"of (epoch based) total latency)",
            extra={"sram_method": "bytes"},
        )

        # Total Excess SRAM Read & Write Time using SRAM Accesses
        logging.debug(
            f"  Total Excess SRAM Latency: "
            f"{format_time(total_excess_sram_duration_accesses_ns)} "
            f"({format_percent(total_excess_sram_duration_accesses_ns, total_duration_accesses_ns,)} "
            f"of (epoch based) total latency)",
            extra={"sram_method": "accesses"},
        )

        # Total Excess SIMD Time
        logging.debug(
            f"  Total Excess SIMD Latency: {format_time(total_excess_simd_duration_ns)} "
            f"({format_percent(total_excess_simd_duration_ns, total_duration_ns)} "
            f"of (epoch based) total latency)",
            extra={"sram_method": "bytes"},
        )
        logging.debug(
            f"  Total Excess SIMD Latency: {format_time(total_excess_simd_duration_accesses_ns)} "
            f"({format_percent(total_excess_simd_duration_accesses_ns, total_duration_accesses_ns)} "
            f"of (epoch based) total latency)",
            extra={"sram_method": "accesses"},
        )

        logging.debug("Per-tile ACE Operations and MACs:")
        total_ace_ops = 0
        total_macs = 0
        for ace_name, buckets in ace_counts.items():
            totals = buckets["all"]
            ace_ops = totals["ace_operations"]
            macs = totals["ace_mac_count"]
            logging.debug(
                f"  {ace_name} - "
                f"ACE Operations: {ace_ops:,}, "
                f"MACs: {macs:,}"
            )
            total_ace_ops += ace_ops
            total_macs += macs
        logging.info(f"Total Executed ACE Operations: {total_ace_ops:,}")
        logging.info(f"Total Executed ACE MACs: {total_macs:,}")
        logging.info(
            f"Maximum Theoretical ACE Execution Time (With No Parallelization Using 1 ACE): "
            f"{format_time(total_ace_ops * ACE_DURATION_NS)}"
        )
        logging.info(
            f"Minimum Theoretical ACE Execution Time "
            f"(With Even Parallelization Across {PER_CHIP_ACE_COUNT} ACEs): "
            f"{format_time(total_ace_ops * ACE_DURATION_NS / PER_CHIP_ACE_COUNT)}"
        )

        if total_ace_ops > 0:
            ace_utilization = (
                (total_ace_ops * ACE_DURATION_SEC / PER_CHIP_ACE_COUNT)
                / (maximum_bottleneck_ns / 1e9)
                * 100
            )
            logging.info(
                f"ACE Utilization (Minimum Theoretical Time/Estimated Processing Time): "
                f"{ace_utilization:.2f}%", extra={"sram_method": "bytes"},
            )
            ace_utilization_accesses = (
                (total_ace_ops * ACE_DURATION_SEC / PER_CHIP_ACE_COUNT)
                / (maximum_bottleneck_accesses_ns / 1e9)
                * 100
            )
            logging.info(
                f"ACE Utilization (Minimum Theoretical Time/Estimated Processing Time): "
                f"{ace_utilization_accesses:.2f}%", extra={"sram_method": "accesses"},
            )

        if not npu_perf_data:
            # Placeholder for NPU MACs
            logging.info("Total Estimated MACs Targeting Digital NPU: [TBD]")

        # Log SRAM Byte totals across the entire chip
        logging.info(
            f"Total SRAM Bytes Read Across Chip: {total_bytes_read:,} bytes"
        )
        logging.debug(
            f"  Total SRAM Data Bytes Read Across Chip: {total_data_bytes_read:,} bytes "
            f"({format_percent(total_data_bytes_read, total_bytes_read)} of reads)"
        )
        logging.debug(
            f"  Total SRAM Control Bytes Read Across Chip: {total_control_bytes_read:,} bytes "
            f"({format_percent(total_control_bytes_read, total_bytes_read)} of reads)"
        )
        logging.info(
            f"Total SRAM Bytes Written Across Chip: {total_bytes_written:,} bytes"
        )
        logging.debug(
            f"  Total SRAM Data Bytes Written Across Chip: "
            f"{total_data_bytes_written:,} bytes "
            f"({format_percent(total_data_bytes_written, total_bytes_written)} "
            f"of writes)"
        )
        logging.debug(
            f"  Total SRAM Control Bytes Written Across Chip: "
            f"{total_control_bytes_written:,} bytes "
            f"({format_percent(total_control_bytes_written, total_bytes_written)} "
            f"of writes)"
        )

        # Log SRAM Access totals across the entire chip
        logging.debug(
            f"Total SRAM Accesses Across Chip: "
            f"{total_sram_accesses:,} accesses"
        )
        logging.debug(
            f"Total SRAM Read Accesses Across Chip: "
            f"{total_read_accesses:,} accesses "
            f"({format_percent(total_read_accesses, total_sram_accesses)} of all accesses)"
        )
        logging.debug(
            f"  Total SRAM Data Read Accesses Across Chip: "
            f"{total_data_read_accesses:,} accesses "
            f"({format_percent(total_data_read_accesses, total_read_accesses)} of read accesses)"
        )
        logging.debug(
            f"  Total SRAM Control Read Accesses Across Chip: "
            f"{total_control_read_accesses:,} accesses "
            f"({format_percent(total_control_read_accesses, total_read_accesses)} of read accesses)"
        )
        logging.debug(
            f"Total SRAM Write Accesses Across Chip: "
            f"{total_write_accesses:,} accesses "
            f"({format_percent(total_write_accesses, total_sram_accesses)} of all accesses)"
        )
        logging.debug(
            f"  Total SRAM Data Write Accesses Across Chip: "
            f"{total_data_write_accesses:,} accesses "
            f"({format_percent(total_data_write_accesses, total_write_accesses)} "
            f"of write accesses)"
        )
        logging.debug(
            f"  Total SRAM Control Write Accesses Across Chip: "
            f"{total_control_write_accesses:,} accesses "
            f"({format_percent(total_control_write_accesses, total_write_accesses)} "
            f"of write accesses)"
        )

        # Log per-tile SRAM Total times based on bytes
        # Informational - not used in estimated latency
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "Actual SRAM Read Time per Tile:",
                extra={"sram_method": "bytes"},
            )
            for tile in tiles:
                data_ns = (
                    data_bytes_read_per_tile[tile] / SRAM_BYTES_READ_PER_CYCLE * CLOCK_PERIOD_NS
                )
                control_ns = (
                    control_bytes_read_per_tile[tile] / SRAM_BYTES_READ_PER_CYCLE * CLOCK_PERIOD_NS
                )
                logging.debug(
                    f"  Tile {tile} - Total SRAM Read Time: "
                    f"{format_time(data_ns + control_ns)}",
                    extra={"sram_method": "bytes"},
                )
                logging.debug(
                    f"    Tile {tile} - Total SRAM Data Read Time: "
                    f"{format_time(data_ns)}",
                    extra={"sram_method": "bytes"},
                )
                logging.debug(
                    f"    Tile {tile} - Total SRAM Control Read Time: "
                    f"{format_time(control_ns)}",
                    extra={"sram_method": "bytes"},
                )

            logging.debug(
                "Actual SRAM Write Time per Tile:",
                extra={"sram_method": "bytes"},
            )
            for tile in tiles:
                data_ns = (
                    data_bytes_written_per_tile[tile] / SRAM_BYTES_WRITTEN_PER_CYCLE * CLOCK_PERIOD_NS
                    if SRAM_BYTES_WRITTEN_PER_CYCLE > 0 else 0
                )
                control_ns = (
                    control_bytes_written_per_tile[tile] / SRAM_BYTES_WRITTEN_PER_CYCLE * CLOCK_PERIOD_NS
                    if SRAM_BYTES_WRITTEN_PER_CYCLE > 0 else 0
                )
                logging.debug(
                    f"  Tile {tile} - Total SRAM Write Time: "
                    f"{format_time(data_ns + control_ns)}",
                    extra={"sram_method": "bytes"},
                )
                logging.debug(
                    f"    Tile {tile} - Total SRAM Data Write Time: "
                    f"{format_time(data_ns)}",
                    extra={"sram_method": "bytes"},
                )
                logging.debug(
                    f"    Tile {tile} - Total SRAM Control Write Time: "
                    f"{format_time(control_ns)}",
                    extra={"sram_method": "bytes"},
                )

        # Log per-tile SRAM Total times based on accesses
        # Informational - not used in estimated latency
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "Actual SRAM Read/Write Time per Tile:",
                extra={"sram_method": "accesses"},
            )
            for tile in tiles:
                total_ns = (
                    (
                        data_read_accesses_per_tile[tile]
                        + control_read_accesses_per_tile[tile]
                        + data_write_accesses_per_tile[tile]
                        + control_write_accesses_per_tile[tile]
                    )
                    / TOTAL_SRAM_ACCESSES_PER_CYCLE * CLOCK_PERIOD_NS
                    if TOTAL_SRAM_ACCESSES_PER_CYCLE > 0 else 0
                )
                logging.debug(
                    f"  Tile {tile} - Total SRAM Read/Write Time: "
                    f"{format_time(total_ns)}",
                    extra={"sram_method": "accesses"},
                )

            logging.debug("Actual SIMD Operation Time per Tile:")
            simd_tiles = sorted(simd_bytes_per_tile.keys())
            for tile in simd_tiles:
                simd_ns = (
                    simd_bytes_per_tile[tile] / SIMD_VECTOR_WIDTH * CLOCK_PERIOD_NS
                )
                logging.debug(
                    f"  Tile {tile} - Total SIMD Operation Time: {format_time(simd_ns)}"
                )

        logging.debug(
            f"Single ACE Tile Die Area: {ACE_TILE_AREA_PS_MM2:.2f} mm^2"
        )
        logging.info(
            "Estimated Die Area to Achieve Estimated Processing Time: "
            f"{ACE_TILE_AREA_PS_MM2 * PER_CHIP_ACE_TILE_COUNT:.2f} mm^2"
        )

    finally:
        with progress_spinner("Closing HDF5 file", spinner_enabled):
            f.close()

    if len(npu_perf_data) > 0:
        logging.info("-------------------------------------------------------------------")
        logging.info("Digital NPU Performance Metrics (from separate ONNX graph processing)")
        logging.info(f"  Total Estimated MACs Targeting Digital NPU: {float(npu_perf_data['macs_in_bn']) * 1e9:,.0f}")
        logging.info(f"  Digital NPU MAC Utilization: {npu_perf_data['mac_utilization']:.2f}%")
        logging.info(f"  Digital NPU Cores: {int(npu_perf_data['cores']):.0f}")
        logging.info(f"  Digital NPU Frequency: {int(npu_perf_data['frequency']) / 1e6:,.0f} MHz")
        logging.info(f"  Digital Estimated Frame Processing Time: {format_time(1e9 / float(npu_perf_data['fps']))}")
        logging.info(f"  Digital Estimated Frame Rate: {npu_perf_data['fps']:,.02f} fps (frames per second)")

        logging.info("-------------------------------------------------------------------")
        frame_latency_ns += 1e9 / float(npu_perf_data["fps"])
        logging.info(f"Combined Analog + Digital NPU Latency: {format_time(frame_latency_ns)}")
        logging.info(f"Combined Analog + Digital NPU Total Estimated Frame Rate: {1e9 / frame_latency_ns:,.2f} "
                     "fps (frames per second)")

    message = (
        "NOTE: The information reported above may include the Analog NPU or the Analog NPU "
        "and a digital NPU as indicated. This estimated processing time and associated frame "
        "rate only covers the execution of the compiled ONNX model(s) running on one or both "
        "of those compute solutions in isolation. It does not include PCIe transfer overhead. "
        "Performance estimation for additional processing steps and a combined compute hardware "
        "solution will be added in future versions."
    )
    logging.info("\n" + textwrap.fill(message, width=80) + "\n")


def main():
    """Define the main entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    # Enable spinner unless we're in DEBUG (too chatty), or yaspin missing / non-tty.
    spinner_enabled = True
    analyze_file(args, spinner_enabled)


if __name__ == "__main__":
    main()
