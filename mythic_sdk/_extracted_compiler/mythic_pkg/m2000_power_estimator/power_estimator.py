# Copyright (C) 2024, Mythic Inc. All rights reserved.
#
"""Estimate the power of the Mythic M2000 chip using an inference L0 Protobuf file."""
from argparse import ArgumentParser, Namespace, ArgumentTypeError
import math
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Dict, Mapping, Tuple, Optional

import numpy as np

from mythic.irs.l0.ir_pb2 import Crate, BaseLauncher, ParameterInfo

MAX_ADC_BITS = 8
MIN_ADC_BITS = 2
MAX_AIDAC_BITS = 8
MIN_AIDAC_BITS = 2
MAX_ACE_INPUTS = 1280
MAX_ACE_OUTPUTS = 272
ACCESSOR_DESCRIPTOR_BYTES = 136
OPERATION_COUNTER_BYTES = 96
OPERATION_COUNTER_WRITEABLE_BYTES = 24
DEFAULT_AVERAGE_TOKEN_LIST_LENGTH = 5.0  # Average number of tokens to aggregate per operation
CLOCK_TREE_POWER_FACTOR = 0.5  # Factor of power from the clock tree over digital logic power
LEAKAGE_POWER_POWER_FACTOR = 0.5   # Factor of power from the leakage power over digital logic power

# Map of PCIe power (mW) by operation mode and number of lanes [ x1, x2, x4 ] from 28nm IP databook
PCIE_POWER = {
    "P0": {1: 127.8, 2: 233.64, 4: 494.64},
    "P0S": {1: 72.36, 2: 122.76, 4: 272.88},
    "P1": {1: 10.71, 2: 19.35, 4: 35.39},
    "P2": {1: 2.466, 2: 3.366, 4: 4.090},
    "POWER_DOWN": {1: 0.288, 2: 0.468, 4: 0.722}
}

# Map of Die-to-Die Power (mW) by operation mode and number of Tx/Rx lanes [ 16Tx/Rx, 12Tx/Rx, 8Tx/Rx ] from 28nm IP databook
D2D_POWER = {
    "IDLE": {16: 71.5, 12: 71.5*0.75, 8: 71.5*0.5},
    "ACTIVE": {16: 214.3, 12: 214.3*0.75, 8: 214.3*0.5},
    "POWER_DOWN": 1.7
}
NUM_D2D_INSTANCES = 4  # Number of Die-to-Die PHY instances in the chip
D2D_ACTIVITY_FACTOR = 0.3  # Estimated activity factor for Die-to-Die PHYs


class EnergyConstants:
    """Holds energy constants for different process nodes to use for estimation."""

    # Energy lookup table (energies in joules per byte or per op, as appropriate)
    ENERGY_TABLE = {
        28: {  # 28nm estimate
            "SRAM_BYTE_READ_ENERGY": 1.0e-12,
            "SRAM_BYTE_WRITE_ENERGY": 1.1e-12,
            "INTERCON_GLOBAL_BYTE_XFER_ENERGY": 1.6e-12,
            "INTERCON_LOCAL_BYTE_XFER_ENERGY": 1.6e-12,
            "ACCESSOR_PROCESSING_ENERGY": 1.0e-12,
        },
        12: {  # 12nm estimate
            "SRAM_BYTE_READ_ENERGY": 0.6e-12,
            "SRAM_BYTE_WRITE_ENERGY": 0.7e-12,
            "INTERCON_GLOBAL_BYTE_XFER_ENERGY": 1.6e-12,  # FF energy is 2fF/toggle so even multiple hops on the interconnect should be negligible
            "INTERCON_LOCAL_BYTE_XFER_ENERGY": 1.6e-12,  # 0.25fJ/um * (0.9V)^2 * 1000um * 8-bits/bytes = 1.6pJ/byte per mm
            "ACCESSOR_PROCESSING_ENERGY": 0.6e-12,
        },
        5: {  # 5nm estimate
            "SRAM_BYTE_READ_ENERGY": 0.5e-12,
            "SRAM_BYTE_WRITE_ENERGY": 0.6e-12,
            "INTERCON_GLOBAL_BYTE_XFER_ENERGY": 1.2e-12,
            "INTERCON_LOCAL_BYTE_XFER_ENERGY": 1.2e-12,
            "ACCESSOR_PROCESSING_ENERGY": 0.6e-12,
        },
    }

    def __init__(self, process_node: int):
        if process_node not in self.ENERGY_TABLE:
            raise ValueError(f"Unsupported process node: {process_node}nm")
        self.consts = self.ENERGY_TABLE[process_node].copy()
        self.consts["TOKEN_UPDATE_ENERGY"] = (
            3 * 8 * (  # 8 bytes per token, 3 accesses (read, modify, write)
                self.consts["SRAM_BYTE_READ_ENERGY"]
                + self.consts["SRAM_BYTE_WRITE_ENERGY"]
            )  # Estimate
        )

    def get(self, key: str) -> float:
        """Retrieve a constant value by its key."""
        return self.consts[key]


@dataclass
class OpEnergy:
    """Energy of an operation."""

    ace_active: float = 0.0
    ace_sleep: float = 0.0
    sram: float = 0.0  # Actual buffer data access
    accessor: float = 0.0
    control: float = 0.0  # Mostly SRAM, but for tokens and descriptors
    noc: float = 0.0

    @property
    def total(self) -> float:
        """Total energy of the operation."""
        return self.ace_active + self.ace_sleep + self.sram + self.accessor + self.control + self.noc

    @property
    def digital(self) -> float:
        """Energy of all digital logic (i.e., not the ACE)."""
        return self.sram + self.accessor + self.control + self.noc

    @property
    def ace(self) -> float:
        """Energy of the ACE including active and sleep."""
        return self.ace_active + self.ace_sleep

    def __add__(self, other: "OpEnergy") -> "OpEnergy":
        """Add two energy profiles."""
        return OpEnergy(
            ace_active=self.ace_active + other.ace_active,
            ace_sleep=self.ace_sleep + other.ace_sleep,
            sram=self.sram + other.sram,
            accessor=self.accessor + other.accessor,
            control=self.control + other.control,
            noc=self.noc + other.noc,
        )

    def __str__(self) -> str:
        """Convert class to string representation of the energy."""
        return (f"    ACE Active: {self.ace_active:8.5f} J\n"
                + f"    ACE Sleep:  {self.ace_sleep:8.5f} J\n"
                + f"    SRAM:       {self.sram:8.5f} J\n"
                + f"    Accessor:   {self.accessor:8.5f} J\n"
                + f"    Control:    {self.control:8.5f} J\n"
                + f"    NOC:        {self.noc:8.5f} J\n"
                + f"    Total:      {self.total:8.5f} J")


class Operation(Enum):
    """Operations in the L0 protobuf."""

    ACE = 1
    COPY = 2
    SIMD = 3
    PAD = 4
    INFEED = 5
    OUTFEED = 6


def get_num_kernel_iterations(iter_spec: ParameterInfo) -> int:  # type: ignore
    """Standalone method to get the number of iterations for an iteration spec."""
    return int(np.prod(iter_spec.num_iterations.dims)
               * np.prod(iter_spec.wq_iterations.dims)
               * iter_spec.stay_count)


def get_num_op_control_iterations(iter_spec: ParameterInfo) -> int:  # type: ignore
    """Standalone method to get the number of iterations for an iteration spec."""
    # THIS IS NOT ACTUALLY IMPLEMENTED IN THE COMPILER YET
    return math.ceil(get_num_kernel_iterations(iter_spec)/4.0)


def get_kernel_bytes(iter_spec: ParameterInfo) -> int:  # type: ignore
    """Standalone method to get the number of bytes in a kernel."""
    return int(np.prod(iter_spec.filter.dims))


def calc_operation_energy(base: BaseLauncher, buffer: ParameterInfo, energy_consts: EnergyConstants, average_token_list_length: float = DEFAULT_AVERAGE_TOKEN_LIST_LENGTH) -> OpEnergy:  # type: ignore
    """Standalone method to calculate the energy of the control signals."""
    energy = OpEnergy()
    n_iterations = get_num_op_control_iterations(buffer.iteration_spec)
    energy.control += n_iterations * energy_consts.get("TOKEN_UPDATE_ENERGY") * average_token_list_length
    energy.control += n_iterations * OPERATION_COUNTER_BYTES * energy_consts.get("SRAM_BYTE_READ_ENERGY") + \
        n_iterations * OPERATION_COUNTER_WRITEABLE_BYTES * energy_consts.get("SRAM_BYTE_WRITE_ENERGY")
    logging.debug(f"Operation {base.elem.name} -- Control: {energy.control:.3g} J")
    return energy


def calc_accessor_energy(base: BaseLauncher, buffer: ParameterInfo, energy_consts: EnergyConstants, write: bool, average_token_list_length: float = DEFAULT_AVERAGE_TOKEN_LIST_LENGTH) -> OpEnergy:  # type: ignore
    """Standalone method to calculate the energy of accessing a parameter."""
    energy = OpEnergy()
    n_bytes = get_kernel_bytes(buffer.iteration_spec)
    n_iterations = get_num_kernel_iterations(buffer.iteration_spec)
    energy.control += n_iterations * energy_consts.get("TOKEN_UPDATE_ENERGY") * average_token_list_length
    energy.accessor += n_iterations * ACCESSOR_DESCRIPTOR_BYTES * (energy_consts.get("SRAM_BYTE_READ_ENERGY") + energy_consts.get("SRAM_BYTE_WRITE_ENERGY"))
    energy.accessor += n_iterations * n_bytes * energy_consts.get("ACCESSOR_PROCESSING_ENERGY")
    energy.sram += n_iterations * n_bytes * (energy_consts.get("SRAM_BYTE_WRITE_ENERGY") if write else energy_consts.get("SRAM_BYTE_READ_ENERGY"))
    logging.debug(f"Accessor {base.elem.name} -- Accessor: {energy.accessor:.3g} J, Control: {energy.control:.3g} J,  SRAM: {energy.sram:.3g} J")
    return energy


# Last updated 2025-01-31 based on ACE_Power_Chart_2025_01_08.xlsx
# This assumes that a lot of operations are run back-to-back so there is no extra calibration time
#
# TODO - Cross-check this estimate against the spreadsheet to match theoretical TOPS/W
def ace_op_energy_time(n_inputs: int, n_outputs: int, n_input_bits: int, n_output_bits: int, sleep: bool) -> Tuple[float, float, float]:
    """Standalone method to calculate the energy of an ACE operation."""
    logging.debug(f"Calculating ACE energy for {n_inputs} inputs, {n_outputs} outputs, {n_input_bits} input bits, and {n_output_bits} output bits.")
    if n_inputs > MAX_ACE_INPUTS or n_inputs < 0:
        raise ValueError(f"Invalid n_inputs: {n_inputs}.")
    if n_outputs > MAX_ACE_OUTPUTS or n_outputs < 0:
        raise ValueError(f"Invalid n_outputs: {n_outputs}.")
    # Calculate currents (in A) estimated at 50C for typical process corner
    i_adc_global = 9.9e-6  # GDAC and biasing
    i_common_mode = 2.2 * 2e-6  # per ADC (assumes pFSR=5, which draws 2.2x common-mode current)
    i_adc_core = 83e-6 + (2 * i_common_mode)  # per ADC
    i_aidac_global_1p0 = 2.65e-4
    i_aidac_global_1p8 = 1.32e-3
    i_aidac_core_1p8 = 11.6e-6  # per AIDAC
    i_aidac_core_1p0 = 4.6e-6  # per AIDAC

    # Inactive AIDACs can be snoozed or be in sleep mode
    # Inactive ADCs can be snoozed or in sleep mode (deep sleep is no longer global)
    if sleep:
        i_adc_core_inactive = 0  # per ADC
        i_aidac_core_inactive_1p0 = 0  # per AIDAC
        i_aidac_core_inactive_1p8 = 0  # per AIDAC
    else:  # Snooze mode
        i_adc_core_inactive = 1.6e-5  # per ADC
        i_aidac_core_inactive_1p0 = 2.4e-6  # per AIDAC
        i_aidac_core_inactive_1p8 = 5.4e-6  # per AIDAC

    # Calculate powergating groups
    input_powergate_groupsize = 1  # ACE supports turning off individual AIDACs
    output_powergate_groupsize = 1  # ACE supports turning off individual ADCs
    n_inputs_active = math.ceil(n_inputs / input_powergate_groupsize) * input_powergate_groupsize
    n_outputs_active = math.ceil(n_outputs / output_powergate_groupsize) * output_powergate_groupsize
    n_inputs_inactive = MAX_ACE_INPUTS - n_inputs_active
    n_outputs_inactive = MAX_ACE_OUTPUTS - n_outputs_active
    # Calculate Power (in W)
    adc_supply_1p0 = 1.0
    aidac_supply_1p0 = 1.0
    aidac_supply_1p8 = 1.8
    active_power = 0.0
    inactive_power = 0.0
    active_power += adc_supply_1p0 * i_adc_global
    active_power += adc_supply_1p0 * i_adc_core * n_outputs_active
    active_power += (aidac_supply_1p0 * i_aidac_global_1p0)
    active_power += (aidac_supply_1p8 * i_aidac_global_1p8)
    active_power += (aidac_supply_1p0 * i_aidac_core_1p0) * n_inputs_active
    active_power += (aidac_supply_1p8 * i_aidac_core_1p8) * n_inputs_active
    inactive_power += adc_supply_1p0 * i_adc_core_inactive * n_outputs_inactive
    inactive_power += aidac_supply_1p0 * i_aidac_core_inactive_1p0 * n_inputs_inactive
    inactive_power += aidac_supply_1p8 * i_aidac_core_inactive_1p8 * n_inputs_inactive
    # ADC calculation time is always 160ns, and we always generate 8-bit outputs
    t_adc = 1e-9 * [0, 160, 160, 160, 160, 160, 160, 160, 160][n_output_bits]
    # Calculate energy
    energy = active_power * t_adc
    inactive_energy = inactive_power * t_adc
    return energy, inactive_energy, t_adc


@dataclass
class M2000_Power():
    """Calculate the power of the M2000 chip for an inference L0 file."""

    l0_pb_path: Path
    inf_rate: int
    digital_process_node: int
    num_aces: int
    num_pcie_lanes: int
    pcie_activity_factor: float
    num_d2d_lanes: int
    num_die_in_system: int
    packet_log_path: Optional[Path] = None
    event_log_path: Optional[Path] = None
    npu_log_path: Optional[Path] = None
    l0_pb: Crate = None  # type: ignore
    packet_log: Optional[Mapping[str, Any]] = None
    event_log: Optional[Mapping[str, Any]] = None
    npu_log: Optional[Mapping[str, Any]] = None
    average_token_list_length: float = DEFAULT_AVERAGE_TOKEN_LIST_LENGTH
    op_energy: Dict[Operation, OpEnergy] = field(default_factory=dict)
    energy_consts: Optional[EnergyConstants] = None

    def __post_init__(self):
        """Load the L0 protobuf and if specified, the packet log."""
        if not self.l0_pb_path.exists():
            raise FileNotFoundError(f"File {self.l0_pb_path} does not exist.")
        if not self.inf_rate > 0:
            raise ValueError(f"Invalid inference rate {self.inf_rate}. Must be positive.")
        self.l0_pb = Crate()
        self.l0_pb.ParseFromString(self.l0_pb_path.read_bytes())

        if self.packet_log_path:
            if self.packet_log_path.exists():
                with open(self.packet_log_path, encoding='utf-8') as packet_log_file:
                    self.packet_log = json.load(packet_log_file)
            else:
                raise FileNotFoundError(f"File {self.packet_log_path} does not exist.")

        if self.event_log_path:
            if self.event_log_path.exists():
                with open(self.event_log_path, encoding='utf-8') as event_log_file:
                    self.event_log = json.load(event_log_file)
                # Update average token list length from event log
                self.average_token_list_length = self.event_log["Token List Entries"]["average"]
            else:
                raise FileNotFoundError(f"File {self.event_log_path} does not exist.")
        logging.debug(f"Using average token list length of {self.average_token_list_length:.1f} entries\n")

        if self.npu_log_path:
            if self.npu_log_path.exists():
                with open(self.npu_log_path, encoding='utf-8') as npu_log_file:
                    self.npu_log = json.load(npu_log_file)
            else:
                raise FileNotFoundError(f"File {self.npu_log_path} does not exist.")
            try:
                self.digital_npu_cores = int(self.npu_log["system_config"]["sys"]["nmps"])
            except Exception as e:
                raise Exception(f"Failed to access expected JSON key/value from {self.npu_log_path}") from e
        else:
            self.digital_npu_cores = 0

        self.energy_consts = EnergyConstants(self.digital_process_node)

    def calc_energy_ace(self) -> OpEnergy:
        """Parse the ACE operations and calculate their power."""
        ace_ops = [launcher.mma_dot
                   for launcher in self.l0_pb.launchers
                   if launcher.mma_dot.base.elem.name != '']
        logging.debug(f"Found {len(ace_ops)} ACE Dot operations.")
        profile = OpEnergy()
        total_time = 0.0
        for op in ace_ops:
            # ACE Operation
            n_iterations = get_num_kernel_iterations(op.input.iteration_spec)
            active_energy, inactive_energy, op_time = ace_op_energy_time(
                n_inputs=1 + np.prod(op.input.iteration_spec.wq_sub_filter.dims),
                n_outputs=np.prod(op.output.iteration_spec.wq_sub_filter.dims),
                n_input_bits=1 + op.input_end_bit - op.input_start_bit,
                n_output_bits=op.adc_cycles.dims[0],
                sleep=True,)
            profile.ace_active += active_energy * n_iterations
            profile.ace_sleep += inactive_energy * n_iterations
            total_time += op_time * n_iterations
            # Digital Energy
            profile += calc_accessor_energy(op.base, op.input, self.energy_consts, write=False, average_token_list_length=self.average_token_list_length)
            profile += calc_accessor_energy(op.base, op.output, self.energy_consts, write=True, average_token_list_length=self.average_token_list_length)
            profile += calc_operation_energy(op.base, op.input, self.energy_consts, average_token_list_length=self.average_token_list_length)
            # TODO - ADD NOC ENERGY

        # Sleep energy - NOT HANDLING SNOOZE VS SLEEP WELL
        needed_sleep_time = self.num_aces / self.inf_rate - total_time
        sleep_energy, sleep_sleep_energy, sleep_time = ace_op_energy_time(
            n_inputs=0,
            n_outputs=0,
            n_input_bits=8,
            n_output_bits=8,
            sleep=True,)
        profile.ace_sleep += (sleep_energy+sleep_sleep_energy) * needed_sleep_time / sleep_time
        # Total power
        self.op_energy[Operation.ACE] = profile
        logging.debug(f"ACE Total -- Energy per Inference: {profile.total:.3g} J,  Power: {profile.total*self.inf_rate:.3g} W\n")
        return profile

    def calc_energy_copy(self) -> OpEnergy:
        """Parse the copy operations and calculate their power."""
        copy_ops = [launcher.copy
                    for launcher in self.l0_pb.launchers
                    if launcher.copy.base.elem.name != '']
        logging.debug(f"Found {len(copy_ops)} Copy operations.")
        profile = OpEnergy()
        for op in copy_ops:
            profile += calc_accessor_energy(op.base, op.input, self.energy_consts, write=False, average_token_list_length=self.average_token_list_length)
            profile += calc_accessor_energy(op.base, op.output, self.energy_consts, write=True, average_token_list_length=self.average_token_list_length)
            profile += calc_operation_energy(op.base, op.input, self.energy_consts, average_token_list_length=self.average_token_list_length)
            # TODO - ADD NOC ENERGY
        # Total power
        self.op_energy[Operation.COPY] = profile
        logging.debug(f"Copy Total -- Energy per Inference: {profile.total:.3g} J,  Power: {profile.total*self.inf_rate:.3g} W\n")
        return profile

    def calc_energy_simd(self) -> OpEnergy:
        """Parse the SIMD operations and calculate their power."""
        simd_ops = [launcher.salu
                    for launcher in self.l0_pb.launchers
                    if launcher.salu.base.elem.name != '']
        logging.debug(f"Found {len(simd_ops)} SIMD operations.")
        profile = OpEnergy()
        for op in simd_ops:
            profile += calc_accessor_energy(op.base, op.input, self.energy_consts, write=False, average_token_list_length=self.average_token_list_length)
            profile += calc_accessor_energy(op.base, op.output, self.energy_consts, write=True, average_token_list_length=self.average_token_list_length)
            profile += calc_operation_energy(op.base, op.input, self.energy_consts, average_token_list_length=self.average_token_list_length)
            # TODO - ADD NOC ENERGY
        # Total power
        self.op_energy[Operation.SIMD] = profile
        logging.debug(f"SIMD Total -- Energy per Inference: {profile.total:.3g} J,  Power: {profile.total*self.inf_rate:.3g} W\n")
        return profile

    def calc_energy_pad(self) -> OpEnergy:
        """Parse the PAD operations and calculate their power."""
        pad_ops = [launcher.pad
                   for launcher in self.l0_pb.launchers
                   if launcher.pad.base.elem.name != '']
        logging.debug(f"Found {len(pad_ops)} Pad operations.")
        profile = OpEnergy()
        for op in pad_ops:
            profile += calc_accessor_energy(op.base, op.output, self.energy_consts, write=True, average_token_list_length=self.average_token_list_length)
            profile += calc_operation_energy(op.base, op.output, self.energy_consts, average_token_list_length=self.average_token_list_length)
            # TODO - ADD NOC ENERGY
        # Total power
        self.op_energy[Operation.PAD] = profile
        logging.debug(f"Pad Total -- Energy per Inference: {profile.total:.3g} J,  Power: {profile.total*self.inf_rate:.3g} W\n")
        return profile

    def calc_energy_infeed(self) -> OpEnergy:
        """Parse the infeed operations and calculate their power."""
        infeed_ops = [launcher.infeed
                      for launcher in self.l0_pb.launchers
                      if launcher.infeed.base.elem.name != '']
        logging.debug(f"Found {len(infeed_ops)} Infeed operations.")
        profile = OpEnergy()
        for op in infeed_ops:
            profile += calc_accessor_energy(op.base, op.output, self.energy_consts, write=True, average_token_list_length=self.average_token_list_length)
            profile += calc_operation_energy(op.base, op.output, self.energy_consts, average_token_list_length=self.average_token_list_length)
            # TODO - ADD NOC ENERGY
        # Total power
        self.op_energy[Operation.INFEED] = profile
        logging.debug(f"Infeed Total -- Energy per Inference: {profile.total:.3g} J,  Power: {profile.total*self.inf_rate:.3g} W\n")
        return profile

    def calc_energy_outfeed(self) -> OpEnergy:
        """Parse the outfeed operations and calculate their power."""
        outfeed_ops = [launcher.outfeed
                       for launcher in self.l0_pb.launchers
                       if launcher.outfeed.base.elem.name != '']
        logging.debug(f"Found {len(outfeed_ops)} Outfeed operations.")
        profile = OpEnergy()
        for op in outfeed_ops:
            profile += calc_accessor_energy(op.base, op.input, self.energy_consts, write=False, average_token_list_length=self.average_token_list_length)
            profile += calc_operation_energy(op.base, op.input, self.energy_consts, average_token_list_length=self.average_token_list_length)
            # TODO - ADD NOC ENERGY
        # Total power
        self.op_energy[Operation.OUTFEED] = profile
        logging.debug(f"Outfeed Total -- Energy per Inference: {profile.total:.3g} J,  Power: {profile.total*self.inf_rate:.3g} W\n")
        return profile

    def calc_energy_interconnect(self) -> float:
        """Parse the packet log and calculate global and local interconnect (NOC) power."""
        if not hasattr(self, "packet_log"):
            logging.warning("Interconnect Total -- No packet log provided. Skipping interconnect energy calculation.\n")
            return 0.0

        global_stats = {}
        local_stats = {}
        total_global_energy = 0.0
        total_local_energy = 0.0

        for key, value in self.packet_log.items():
            global_traffic = value.get("inter_tile_traffic", {})
            global_bytes_transferred = global_traffic.get("bytes_transferred", 0)
            global_energy = global_bytes_transferred * self.energy_consts.get("INTERCON_GLOBAL_BYTE_XFER_ENERGY")
            global_stats[key] = {
                "bytes_transferred": global_bytes_transferred,
                "energy_joules": global_energy
            }
            total_global_energy += global_energy

            local_bytes_transferred = value.get("total_bytes_transferred", 0) - global_bytes_transferred
            local_energy = local_bytes_transferred * self.energy_consts.get("INTERCON_LOCAL_BYTE_XFER_ENERGY")
            local_stats[key] = {
                "bytes_transferred": local_bytes_transferred,
                "energy_joules": local_energy
            }
            total_local_energy += local_energy

        total_energy = total_global_energy + total_local_energy

        # Print the results
        for k, stats in global_stats.items():
            logging.debug(f"Global {k}: {stats['bytes_transferred']} bytes, estimated energy = {stats['energy_joules']:.2e} J")
        for k, stats in local_stats.items():
            logging.debug(f"Local {k}: {stats['bytes_transferred']} bytes, estimated energy = {stats['energy_joules']:.2e} J")

        logging.debug(f"Interconnect Global -- Energy per Inference: {total_global_energy:.3g} J,  Power: {total_global_energy*self.inf_rate:.3g} W")
        logging.debug(f"Interconnect Local -- Energy per Inference: {total_local_energy:.3g} J,  Power: {total_local_energy*self.inf_rate:.3g} W")
        logging.debug(f"Interconnect Total -- Energy per Inference: {total_energy:.3g} J,  Power: {total_energy*self.inf_rate:.3g} W\n")

        return total_energy

    def calc_pcie_power(self) -> float:
        """Calculate the power of the PCIe PHY based on system topology."""
        if self.num_pcie_lanes not in PCIE_POWER["P0"]:
            raise ValueError(f"Invalid number of PCIe lanes: {self.num_pcie_lanes}.")

        active_power = PCIE_POWER["P0"][self.num_pcie_lanes] * self.pcie_activity_factor
        idle_power = PCIE_POWER["P1"][self.num_pcie_lanes] * (1 - self.pcie_activity_factor)
        power = active_power + idle_power
        logging.debug(f"PCIe PHY Power: {power:.3g} mW")
        return power / 1000  # Convert to W

    def calc_d2d_power(self) -> float:
        """Calculate the power of the Die-to-Die PHY based on system topology."""
        if self.num_die_in_system < 1 or self.num_die_in_system > 5:
            raise ValueError(f"Invalid number of M2000 die: {self.num_die_in_system}.")

        if self.num_die_in_system > 1:
            if self.num_d2d_lanes not in D2D_POWER["IDLE"]:
                raise ValueError(f"Invalid number of Die-to-Die lanes: {self.num_d2d_lanes}.")

        # Default to power down
        power = D2D_POWER["POWER_DOWN"] * NUM_D2D_INSTANCES

        # Calculate active and idle power for topologies with more than one die
        if self.num_die_in_system > 1:
            num_active_phy = self.num_die_in_system - 1
            active_power = D2D_POWER["ACTIVE"][self.num_d2d_lanes] * D2D_ACTIVITY_FACTOR * num_active_phy
            idle_power = D2D_POWER["IDLE"][self.num_d2d_lanes] * (1 - D2D_ACTIVITY_FACTOR) * num_active_phy
            active_phy_power = (active_power + idle_power) * num_active_phy

            num_inactive_phy = NUM_D2D_INSTANCES - num_active_phy
            inactive_phy_power = D2D_POWER["POWER_DOWN"] * num_inactive_phy

            power = active_phy_power + inactive_phy_power

            logging.debug(f"Die-to-Die Number of Active PHY: {num_active_phy}")
            logging.debug(f"Die-to-Die PHY Active Power: {active_power:.3g} mW")
            logging.debug(f"Die-to-Die PHY Idle Power: {idle_power:.3g} mW")
            logging.debug(f"Die-to-Die Total Active PHY Power: {active_phy_power:.3g} mW")
            logging.debug(f"Die-to-Die Number of Inactive PHY: {num_inactive_phy}")
            logging.debug(f"Die-to-Die Total Inactive PHY Power: {inactive_phy_power:.3g} mW")

        logging.debug(f"Die-to-Die PHY Power: {power:.3g} mW\n")
        return power / 1000  # Convert to W

    def calc_energy_digital_npu(self) -> float:
        """Calculate the power of the digital NPU."""
        try:
            # Currently, this just pulls the mW value from the profiling JSON.
            power = float(self.npu_log["profiling"]["power_mw_5nm_typ"])
        except Exception as e:
            raise Exception("Unable to retrieve 'power_mw_5nm_typ' from digital npu log!") from e
        try:
            # Get the fps value to translate power/energy to per-frame.
            fps = float(self.npu_log["profiling"]["fps"])
        except Exception as e:
            raise Exception("Unable to retrieve 'fps' from digital npu log!") from e

        return (power / fps) / 1000  # convert to W per frame

    def calc_power(self) -> float:
        """Calculate and print the power."""
        # Calculate and print the ACE op energies
        ace_op_energies = {i: sum(ace_op_energy_time(
                           n_input_bits=i,
                           n_output_bits=i,
                           n_inputs=MAX_ACE_INPUTS,
                           n_outputs=MAX_ACE_OUTPUTS,
                           sleep=False,
                           )[0:1]) for i in reversed(range(MIN_ADC_BITS, MAX_ADC_BITS+1))}
        ace_energy_table = "\n".join([f"    {b}b: {e*1e9:.3g} nJ   {2*MAX_ACE_INPUTS*MAX_ACE_OUTPUTS/e/1e12:.3g} TOPS/W" for b, e in ace_op_energies.items()])
        logging.debug(f"ACE Op Energies:\n{ace_energy_table}\n")
        # Get the energy for each unit
        total_energy = OpEnergy()
        total_energy += self.calc_energy_ace()
        total_energy += self.calc_energy_copy()
        total_energy += self.calc_energy_simd()
        total_energy += self.calc_energy_pad()
        total_energy += self.calc_energy_infeed()
        total_energy += self.calc_energy_outfeed()
        operation_power = total_energy.total*self.inf_rate
        interconnect_power = self.calc_energy_interconnect()*self.inf_rate
        # TODO - Add leakage, clock tree and chip I/O power
        # leakage_power = LEAKAGE_POWER_POWER_FACTOR * total_energy.digital * self.inf_rate
        # clock_tree_power = CLOCK_TREE_POWER_FACTOR * total_energy.digital * self.inf_rate
        # pcie_power = self.calc_pcie_power()
        # d2d_power = self.calc_d2d_power()
        total_power = operation_power + interconnect_power

        if self.npu_log:
            # Keep this separate for now.
            digital_npu_power = self.calc_energy_digital_npu() * self.inf_rate

        # Report the calculated power metrics
        logging.debug("Active Energy Breakdown: (per inference)")
        logging.debug(total_energy)
        logging.debug("")
        logging.info(f"Process Nodes:  Analog 28nm, Digital {self.digital_process_node}nm")
        logging.info(f"Number of ACEs: {self.num_aces}")
        if self.npu_log:
            logging.info(f"Number of Digital NPU Cores: {self.digital_npu_cores}")
        logging.info(f"Inference Rate: {self.inf_rate} frames/second (fps)")
        logging.info("")
        logging.info(f"Analog NPU Estimated Power: {total_power:5.3f} W for target ONNX file")
        logging.info(f"    Functional Unit Power: {operation_power:5.3f} W")
        logging.info(f"    Interconnect Power:    {interconnect_power:5.3f} W")
        # TODO - Add leakage, clock tree and chip I/O power
        # logging.info(f"    Leakage Power:         * W")
        # logging.info(f"    Clock Tree Power:      * W")
        # logging.info(f"    PCIe Power:            * W")
        # logging.info(f"    Die-to-Die Power:      * W")
        if self.npu_log:
            logging.info("")
            logging.info(f"Digital NPU Estimated Power: {digital_npu_power:5.3f} W for digital target ONNX file")
            logging.info("")
            logging.info(f"Total Combined (Analog + Digital NPU) Power: {total_power + digital_npu_power:5.3f} W")
        logging.info("")
        logging.info("NOTE: Estimation for leakage, clock tree, chip I/O power will be added in future versions.")
        return total_power


def multiple_of_4(value: str) -> int:
    """Ensure value is a multiple of 4."""
    ivalue = int(value)
    if ivalue % 4 != 0:
        raise ArgumentTypeError(f"{value} is not a multiple of 4")
    return ivalue


def parse_args() -> Namespace:
    """Parse and return command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--l0-pb-path", type=Path, dest="l0_pb_path", required=True, help="Path to the L0 protobuf file")
    parser.add_argument("--inf-rate", type=int, default=30, dest="inf_rate", help="Inference rate in frames/second (fps)")
    parser.add_argument("--digital-process-node", type=int, default=5, dest="digital_process_node", help="Digital process node in nm (5 or 12 or 28)")
    parser.add_argument("--num-aces", type=multiple_of_4, default=24, dest="num_aces", help="Number of ACEs per chip in integer multiples of 4 (default: %(default)s)")
    parser.add_argument("--num-pcie-lanes", type=int, default=4, dest="num_pcie_lanes", help="Number of PCIe lanes (1, 2 or 4)")
    parser.add_argument("--pcie-activity-factor", type=float, default=0.3, dest="pcie_activity_factor", help="PCIe activity factor [0,1]")
    parser.add_argument("--num-d2d-lanes", type=int, default=12, dest="num_d2d_lanes", help="Number of Die-to-Die lanes per PHY (8, 12 or 16)")
    parser.add_argument("--num-die-in-system", type=int, default=1, dest="num_die_in_system", help="Number of M2000 die in system topology (1-5)")
    parser.add_argument("--packet-log-path", type=Path, dest="packet_log_path", help="Path to the packet log file (JSON) to estimate interconnect power")
    parser.add_argument("--event-log-path", type=Path, dest="event_log_path", help="Path to the event log file (JSON) with event data")
    parser.add_argument("--npu-log-path", type=Path, dest="npu_log_path", help="Path to the digital NPU log file (JSON) with power data")
    parser.add_argument("--log-level", type=str, default="INFO", dest="log_level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    return parser.parse_args()


def main():
    """Run the main function."""
    args = parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid logging level: {args.log_level}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    logging.basicConfig(level=log_level, format="%(message)s")
    del args.log_level  # Remove log level from args

    m2000_power = M2000_Power(**vars(args))
    m2000_power.calc_power()


if __name__ == "__main__":
    main()
