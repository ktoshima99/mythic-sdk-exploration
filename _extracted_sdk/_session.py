# noqa-flake8-docstrings
from contextlib import contextmanager
import os
import logging
from enum import Enum, unique
from onnx import shape_inference, checker

from munc import ops, _verify
from munc._session_tools import _collect_debugging_point, get_model_type
from munc._change_opset import change_opset
from munc._constants import (DEBUG_DIR, MUNC_INTERNAL_ONNX_OPSET, MODELType, DEFAULT_DSF_PARAMETER_GROUP,
                             HardwareType, COMPILER_ONNX_OPSET)
from munc import _loader_tools
from munc._stats_collector import StatsCollector
from munc.hw_specs import hw_config_registry
from munc._torchnet import TorchNet, ActivationCkptConfig
from munc.op_config import get_op_debug, op_conf, op_conf_seq, op_do, configure_ops, instantiate_op, nop
from munc.bcm.bcm_models import digitalmodel
from munc._base_op import OpContext


@unique
class ADD_SCALING_TYPE(str, Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    NONE = 'None'
    INPUT_AND_OUTPUT = 'input_and_output'


logger = logging.getLogger(__name__)


class Session(OpContext):
    def __init__(self,
                 model,
                 stats=None,
                 verbose=True,
                 loader=None,
                 stat_clipping_percentile=0,
                 stat_n_samples_default=5000,
                 stat_shuffle=True,
                 device_name=None,
                 qat=False,
                 debug=None,
                 node_filter=lambda node: True,
                 torchnet_layer_factory=None,
                 activation_ckpt_config: ActivationCkptConfig | None = None,
                 ):
        """Initialize the session.

        Parameters
        ----------
            model : ONNXModel
                The ONNX model to use.
            stats : StatsCollector, optional
                An existing statistics collector. If not provided, the session will create one using the
                `loader`, `stat_clipping_percentile`, `stat_n_samples_default`, and `device_name` parameters.
            verbose : bool, optional
                Enable logging. Defaults to True.
            loader : torch.utils.data.DataLoader or generator, optional
                Dataloader for the input model. Each batch must be a sequence of torch.Tensor objects, one per
                model input, in the same order as the ONNX model inputs. If None, a dummy dataloader is created via
                `munc._loader_tools.dummy_loader`.
            stat_clipping_percentile : float
                Clipping percentile to compute.
            stat_n_samples_default : int, optional
                Default number of samples to use for statistics. Defaults to 5000.
            stat_shuffle : bool, optional
                Whether to shuffle the dummy dataloader. Defaults to True.
            device_name : str, optional
                Device to run inference on (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
            qat : bool, optional
                Enable quantization-aware training for off-chip Conv/Gemm operations. Defaults to False.
            debug : int, optional
                Number of samples to use for generating debugging plots. Defaults to the value of the
                DEBUG_N_SAMPLES environment variable, or 100 if not set.
           node_filter : Optional[Callable[[ONNXNode], bool]]
                Apply ops only to nodes that satisfy this predicate. By default, all nodes are accepted.
           torchnet_layer_factory : Optional[Callable]
                TorchNet layer factory for the model. Passed through to TorchNet. Defaults to None.
            activation_ckpt_config : ActivationCkptConfig, optional
                Per-layer activation checkpointing config. Passed through to TorchNet.
        """
        self.verbose = verbose
        self._model = model
        self.debug_n_samples = int(os.environ.get('DEBUG_N_SAMPLES', 100)) if debug is None else debug
        self._torchnet_layer_factory = torchnet_layer_factory
        self._activation_ckpt_config = activation_ckpt_config
        self.n_samples_default = stat_n_samples_default
        self.device_name = device_name
        self._data_loader = _loader_tools.dummy_loader(model, shuffle=stat_shuffle) if loader is None else loader
        if stats:
            self._stats = stats
        else:
            self._stats = StatsCollector(model=model,
                                         loader=self.data_loader,
                                         clipping_percentile=stat_clipping_percentile,
                                         n_samples_default=self.n_samples_default,
                                         device_name=self.device_name,
                                         shuffle=stat_shuffle,
                                         layer_factory=self.torchnet_layer_factory)

        self.qat = qat
        self.node_filter = node_filter

    @property
    def model(self):
        """Return an ONNX model associated with this session."""
        return self._model

    @property
    def hwconfig(self):
        """Return a hardware configuration associated with this session."""
        return self.model.hwconfig or hw_config_registry[HardwareType.BOREAS]

    @property
    def stats(self):
        """Return a StatsCollector associated with this session."""
        return self._stats

    @property
    def data_loader(self):
        """Return a dataloader associated with this session."""
        return self._data_loader

    @property
    def torchnet_layer_factory(self):
        """Return a TorchNet layer factory associated with this session."""
        return self._torchnet_layer_factory

    def get_copy_of_stats(self):
        """
        Get a copy of statistics.

        Returns
        -------
        dict
            A deep copy of statistics dict which contains output node names as keys and a dict of statistics as values.
            The dict of statistics contains the statistic names as keys and the corresponding values.
        """
        return self.stats.get_copy_of_stats()

    def get_copy_of_per_epoch_stats(self):
        """
        Get a copy of per epoch statistics.

        Returns
        -------
        dict
            A deep copy of per epoch statistics dict which contains output node names as keys and a dict of statistics
            as values. The dict of statistics contains the statistic names as keys and the corresponding values.
            Currently, per epoch statistics are calculated if per_channel flag is set to True while collecting
            statistics.
        """
        return self.stats.get_copy_of_per_epoch_stats()

    # Collect statistics
    def collect_stats(self, N_samples_min=None):
        self._stats.collect(n_samples_min=N_samples_min)

    # Run op
    def run_all_nodes(self, op):
        debug_flags = get_op_debug(op)
        op = instantiate_op(op, self.verbose)
        if op is None:
            # The op is disabled
            return
        op.configure(context=self, verbose=self.verbose,
                     # Only apply this op to a match (sequence of nodes selected by the op's pattern) if
                     # our `node_filter` accepts all the matched nodes.
                     include_pred=lambda match: all(self.node_filter(node) for node in match))
        if "plot_before" in debug_flags:
            self._create_debugging_plots(debug_flags["plot_before"])
        if debug_flags.get("stop"):
            breakpoint()
        op.run_op()
        if "plot" in debug_flags:
            self._create_debugging_plots(debug_flags["plot"])

    def run_ops(self, *ops):
        for op in ops:
            self.run_all_nodes(op)

    def _get_process_original_graph_ops(self):
        def check_model():
            # Will check if the number of input/outputs are correct and that attributes are valid
            model_proto = shape_inference.infer_shapes(self.model._model_proto)
            checker.check_model(model_proto, full_check=True)

        return op_conf_seq(
            *self.get_change_opset_ops(),
            # RemoveDanglingNodes is needed because `change_opset` may result in dangling nodes.
            ops.RemoveDanglingNodes,
            ops.RemoveShapeInferenceNodes,
            ops.GeneralizeBatchSize,
            ops.PostBatchNormFolding,
            ops.PreBatchNormFolding,
            ops.InferStoreTensorShapes,
            ops.MarkUnsupportedOpsOffChip,
            ops.AddNetworkOutputsForIntermediateNodes,
            ops.FixDefaultResizeROI,
            ops.RenameNodesAndEdges,

            op_do(self.stats.reset_stats),
            op_do(check_model)
        )

    def _process_original_graph(self, nodes_with_aux_network_outputs=()):
        ops_to_run = self._get_process_original_graph_ops()
        config = [
            op_conf(ops.AddNetworkOutputsForIntermediateNodes, node_name_list=nodes_with_aux_network_outputs)
        ]
        self.run_ops(*configure_ops(ops_to_run, config))

    def get_original_to_mythic_conversion_ops(self,
                                              scale_offchip_nodes=False,
                                              scale_concat_inputs=False,
                                              optimize_wsf=False,
                                              hardware_config_name=HardwareType.BOREAS):
        hwconfig = hw_config_registry[hardware_config_name]

        def set_hwconfig_metadata():
            logger.info(f"Setting hardware type of the training model to {hwconfig.name}")
            self.model.hwconfig = hwconfig

        return op_conf_seq(
            op_do(lambda: assert_model_is(self._model, MODELType.ORIGINAL)),
            op_do(set_hwconfig_metadata),
            *self._get_process_original_graph_ops(),
            nop("BeforeConversionToMythic", plot="original"),

            ops.ConvertMatMulToGemm,
            ops.ReplaceSiluPatternWithNode,
            op_conf(ops.CloneConvWeights, max_copies=1),
            op_do(self.stats.reset_stats),  # This is only needed if CloneConvWeights made changes.

            # This op has to be before AddInputShifting, which uses sign marking.
            op_conf(ops.MarkSignedNodes, enabled=hwconfig.signed),

            ops.AddInputShifting,
            ops.AddInputScaling,
            ops.AddOutputScaling,

            op_conf(ops.MoveLastBiasOffChip, enabled=False),
            op_do(self.stats.reset_stats),  # This is only needed if MoveLastBiasOffChip made changes.

            # Nick: trainable=True should improve accuracy, but caused a crash in Yolov5s6u.
            op_conf(ops.AddOnOffChipTransitionScaling, enabled=not scale_offchip_nodes),

            # Here we need to mark signed nodes again, because the previous ops could add new `add` or `mul` nodes.
            op_conf(ops.MarkSignedNodes, enabled=hwconfig.signed),

            op_conf(ops.MarkQATNodes, self.qat),

            ops.MarkDepthwiseConvsAsDigital,

            ops.InsertReluForPositiveValues,
            ops.MoveReLUBeforeMaxPoolOp,

            op_conf(ops.SplitInputs, split_bias_fp=True),

            op_conf(ops.FuseAddToSum, enabled=False),

            ops.InjectInputClippings,
            ops.ConvGemmWeightScaling,

            # Randomize weights (if needed)
            op_conf(ops.RandomizeWeightsAndBiases, alpha=0.8, enabled=False),

            # Inject scaling cancellation nodes - these must always be present
            ops.InjectScalingOnMulInputs,
            ops.InjectScalingOnMatMulInputs,
            ops.InjectScalingOnSoftmaxInputs,

            # Inject scaling nodes which will get absorbed into the Mythic nodes
            # and therefore must always be present
            ops.InjectScalingOnAddOutput,
            ops.InjectScalingOnMatMulOutput,
            ops.InjectScalingOnMulOutput,
            ops.InjectScalingOnSoftmaxOutput,

            # Then, inject scaling nodes which will not get absorbed (and therefore are not needed if
            # another scaling node is already present). These ops should check for the presence of
            # a scaling node on the same edge before injecting a new one.
            ops.InjectScalingOnAddInputs,
            op_conf(ops.InjectScalingOnConcatInputs, enabled=scale_concat_inputs),
            op_conf(ops.InferStoreTensorShapes, enabled=scale_concat_inputs),
            op_conf(ops.EqualizeConcatInputs, enabled=scale_concat_inputs),

            # Scaling ops
            ops.BreakCompositeScaleIntoFSRAndDigitalScales,
            op_conf(ops.ScaleAllNodes, debug=self.debug_n_samples, scale_offchip_nodes=scale_offchip_nodes),
            op_conf(ops.BreakFSRIntoPFSRAndIFSR, clip_weights=not optimize_wsf),

            # An empty range is specified here to have the op disabled by default.
            op_conf(ops.RemoveMulByOne, min_val=1.0, max_val=-1.0),
            op_conf(ops.BreakDigitalScalesIntoFactors, break_into_attrs=True),

            # Converts unsupported activations to supported ones
            ops.AutoNameNodes,
            ops.ConvertUnsupportedToSupportedActivations,

            # Diagnostic ops
            ops.InferStoreTensorShapes,

            # Add input shift on-chip to prevent negative MMA inputs
            ops.CreateActivationCompensation,
            ops.AbsorbActivationShift,

            ops.InjectMulClipping,
            op_conf(ops.GroupMMAOps, group_activations=True),
            op_conf(ops.GroupAddOutputOps, group_mul_nodes=False),
            ops.GroupSoftmaxScalingNodes,
            ops.GroupMulMatMulScalingNodes,
            op_conf(ops.ConvertNodesToMythic),
            ops.GroupSumQMul,
            ops.AdjustMythicSumActivation,

            # Rescale off-chip nodes to the same range as on-chip nodes
            ops.RenormalizeOffchipNodes,

            op_conf(ops.OptimizeWSF, wsf_log_file=DEBUG_DIR / "scaling/wsf_scan.json", enabled=optimize_wsf,
                    debug_flags=dict(plot_before="no_opt_mythic")),
            op_conf(ops.ReduceADCClipping, enabled=False, input_patterns=None, pctl=0.9999),
            op_conf(ops.OptimizeIFSR, enabled=False, force_per_channel=True),
            op_conf(ops.MakeDSFsTrainable),
            op_conf(ops.PinLastOnChipConvDSF, enabled=False),
            op_conf(ops.PinLastOnChipMul, enabled=False),

            op_do(lambda: self._model.set_meta_data('__type', MODELType.MYTHIC)),
            nop("AfterConversionToMythic", plot="initial_mythic")
        )

    def convert_original_to_mythic(self,
                                   add_scaling_type=ADD_SCALING_TYPE.INPUT_AND_OUTPUT,
                                   randomize_weights=False,
                                   weight_heuristics_n_sigmas=None,
                                   split_inputs=True,
                                   w_scale=1.0,
                                   nodes_with_aux_network_outputs=(),
                                   mythic_node_map=None,
                                   fuse_add_to_sum=False,
                                   clone_conv_weights_num_times=1,
                                   trainable_dsfs=DEFAULT_DSF_PARAMETER_GROUP,
                                   trainable_last_dsfs=True,
                                   reduce_adc_clipping=False,
                                   scale_offchip_nodes=True,
                                   move_last_bias_offchip=False,
                                   remove_mul_by_one_range=[0.0, -1.0],
                                   scale_concat_inputs=False,
                                   make_on_off_chip_transition_scaling_trainable=False,
                                   break_fsr_into_pfsr_and_ifsr_max_dsf=float('inf'),
                                   break_fsr_into_pfsr_and_ifsr_half_pFSR_arr=None,
                                   break_fsr_into_pfsr_and_ifsr_half_iFSR_arr=None,
                                   optimize_wsf=False,
                                   hardware_config_name=HardwareType.BOREAS):
        ops_to_run = self.get_original_to_mythic_conversion_ops(scale_offchip_nodes=scale_offchip_nodes,
                                                                scale_concat_inputs=scale_concat_inputs,
                                                                optimize_wsf=optimize_wsf,
                                                                hardware_config_name=hardware_config_name)
        config = [
            op_conf(ops.AddNetworkOutputsForIntermediateNodes, node_name_list=nodes_with_aux_network_outputs),
            op_conf(ops.CloneConvWeights, max_copies=clone_conv_weights_num_times),
            op_conf(ops.MoveLastBiasOffChip, enabled=move_last_bias_offchip),
            op_conf(ops.AddOnOffChipTransitionScaling, trainable=make_on_off_chip_transition_scaling_trainable),
            op_conf(ops.SplitInputs, enabled=split_inputs),
            op_conf(ops.FuseAddToSum, enabled=fuse_add_to_sum),
            op_conf(ops.ConvGemmWeightScaling, w_scale=w_scale, num_sigmas=weight_heuristics_n_sigmas,
                    pctl=None if weight_heuristics_n_sigmas else 1.0),
            op_conf(ops.RandomizeWeightsAndBiases, enabled=randomize_weights),
            op_conf(ops.InjectScalingOnAddOutput,
                    enabled=(add_scaling_type in [ADD_SCALING_TYPE.OUTPUT, ADD_SCALING_TYPE.INPUT_AND_OUTPUT])),
            op_conf(ops.InjectScalingOnAddInputs,
                    enabled=(add_scaling_type in [ADD_SCALING_TYPE.INPUT, ADD_SCALING_TYPE.INPUT_AND_OUTPUT])),
            op_conf(ops.BreakFSRIntoPFSRAndIFSR, max_dsf=break_fsr_into_pfsr_and_ifsr_max_dsf,
                    half_pFSR_arr=break_fsr_into_pfsr_and_ifsr_half_pFSR_arr,
                    half_iFSR_arr=break_fsr_into_pfsr_and_ifsr_half_iFSR_arr),
            op_conf(ops.RemoveMulByOne, min_val=remove_mul_by_one_range[0], max_val=remove_mul_by_one_range[1]),
            op_conf(ops.ConvertNodesToMythic, mythic_node_map=mythic_node_map, trainable_dsfs=trainable_dsfs),
            op_conf(ops.ReduceADCClipping, enabled=reduce_adc_clipping),
            op_conf(ops.MakeDSFsTrainable, enabled=trainable_dsfs),
            op_conf(ops.PinLastOnChipConvDSF, enabled=(trainable_dsfs and not trainable_last_dsfs)),
            op_conf(ops.PinLastOnChipMul, enabled=(trainable_dsfs and not trainable_last_dsfs)),
        ]
        self.run_ops(*configure_ops(ops_to_run, config))

    def _create_debugging_plots(self, tag, saving_dir=DEBUG_DIR / "full_range", save_only=False):
        if self.debug_n_samples > 0:
            _collect_debugging_point(tag, self.model, self.stats, self.debug_n_samples, saving_dir=saving_dir,
                                     save_only=save_only)

    def get_mythic_to_bcm_conversion_ops(self):
        """Return ops for converting a MUNC graph with the Mythic Node to BCM IR."""
        return op_conf_seq(
            nop("BeforeConversionToBCM", plot="trained_mythic"),
            ops.UpdateSumMulAttributes,
            ops.ClearTrainableValueList,
            ops.RemoveAllHWFidelityOps,
            ops.RemoveNetworkOutputsFromIntermediateNodes,
            op_conf(ops.InferStoreTensorShapes, update_outputs=True),
            ops.ChannelPaddingTo8,
            op_conf(ops.InferStoreTensorShapes, update_outputs=True),
            ops.ConvertMythicToConvs,
            ops.WeightBiasQuantClip,
            ops.HardCodeWeightAndBias,
            ops.RenameNodesAndEdges,
            ops.BreakBiasIntoRows,
            ops.AddInputShiftingOnchipToHardsigmoid,
            ops.ConvertConvsToBCM,
            ops.ConvertSumsToBCM,
            ops.ConvertParallelTransitionsToChannelwideMul,
            ops.AddCompilerOutputDType,
            ops.AbsorbOffchipMulNodes,
            op_do(lambda: self._model.set_meta_data('__type', MODELType.BCM)),
            nop("AfterConversionToBCM", plot="bcm")
        )

    def convert_mythic_to_bcm(self, bcm_class_str, bcm_attr_str=None,
                              acm_hardware_name=None, acm_noise_name=None,
                              npad=8, auxiliary_outputs_to_delete=()):
        """Convert MUNC graphs with the Mythic Node to BCM IR.

        Parameters
        ----------
        bcm_class_str : str
            Name of the BCM MMA to run inference with.
        bcm_attr_str : str
            Name of the BCM MMA attributes to use. If None then the default is
            used. This supports the `dot` convince. e.g.
            "SimpleAttribute.no_noise"
        acm_hardware_name : str
            The name of the ACM hardware model, by default None
        acm_noise_name : str
            The name of the ACM noise model
        npad : int
            The number of channels to pad the input to
        """
        ops_to_run = self.get_mythic_to_bcm_conversion_ops()
        config = [
            op_conf(ops.RemoveNetworkOutputsFromIntermediateNodes, auxiliary_outputs=auxiliary_outputs_to_delete),
            op_conf(ops.ChannelPaddingTo8, number_of_input_channels=npad),
            op_conf(ops.ConvertConvsToBCM, bcm_class_str=bcm_class_str, bcm_attr_str=bcm_attr_str,
                    acm_hardware_name=acm_hardware_name, acm_noise_name=acm_noise_name),
        ]
        self.run_ops(*configure_ops(ops_to_run, config))

    def get_bcm_to_artifact_conversion_ops(self):
        """Return ops for converting a BCM model to a compiler ready model."""
        return op_conf_seq(
            op_do(lambda: assert_model_is(self.model, MODELType.BCM)),
            op_conf(ops.SwitchBCM, bcm_class_str=digitalmodel.FACTORY_NAME),
            ops.RemoveDummyChannelPadding,
            ops.InferStoreTensorShapes,
            op_conf(ops.AdjustFirstConvForRGBToBGR, enabled=False),
            ops.AttachLUTAttribute,
            op_conf(op_do(lambda: _verify.verify_compiler_model(self.model, self.hwconfig)),
                    name="VerifyCompilerModel"),
            op_do(lambda: self._model.set_meta_data('__type', MODELType.COMPILER)),
            op_do(lambda: self.change_opset(COMPILER_ONNX_OPSET)),
        )

    def get_change_opset_ops(self):
        """Change model's opset.

        Parameters
        ----------
        target_opset_version : int, optional
            A target opset. Defaults to MUNC_INTERNAL_ONNX_OPSET.
        """
        def change_opset_op_func(op, target_opset_version):
            change_opset(op.model, target_opset_version, op.stats, op.verbose)

        return op_conf_seq(
            ops.SanityCheckOffChipMarking,
            ops.AutoNameNodes,
            ops.MakeEdgesUnique,
            ops.ConvertConstsToInitializers,
            ops.RemoveDanglingNodes,
            op_conf(op_do(change_opset_op_func, pass_op=True), name="ChangeOpset",
                    target_opset_version=MUNC_INTERNAL_ONNX_OPSET)
        )

    def change_opset(self, target_opset_version=MUNC_INTERNAL_ONNX_OPSET):
        """Change model's opset.

        Parameters
        ----------
        target_opset_version : int, optional
            A target opset. Defaults to MUNC_INTERNAL_ONNX_OPSET.
        """
        assert get_model_type(self.model) in [MODELType.ORIGINAL, MODELType.COMPILER]
        ops_to_run = self.get_change_opset_ops()
        config = {"ChangeOpset": dict(target_opset_version=target_opset_version)}
        self.run_ops(*configure_ops(ops_to_run, config))

    def make_torch_net(self):
        """Create a TorchNet instance for the ONNX model of specified by this session.

        Before creating a TorchNet instance this method makes sure the model is compatible with
        TorchNet by calling `self.change_opset()`.
        """
        # All the Mythic model type already use the MUNC internal opset, only need to handle FP models here.
        if get_model_type(self.model) == MODELType.ORIGINAL:
            self.change_opset()
        return TorchNet(
            self.model,
            layer_factory=self.torchnet_layer_factory,
            activation_ckpt_config=self._activation_ckpt_config)

    def configuration(self, **settings):
        """Return a context manager that temporary changes some session's settings.

        In the scope of the context manager session attributes specified in `settings` are temporarily changed to
        the values specified in the dictionary.

        Example
        -------
        with sess.configuration(node_filter=lambda node: node in conv_nodes_to_clone):
            sess.run_all_nodes(ops.CloneConvWeights(max_copies=MAX_COPIES, repeat_dim=1))

        Parameters
        ----------
        settings : dict
            A dictionary with the settings to temporarily change. Keys are session's attribute names and values are
            the temporary values.

        Returns
        -------
        ContextManager
            A context manager that temporary modified session settings.
        """
        return _attr_values(self, **settings)


@contextmanager
def _attr_values(obj, **new_attr_values):
    """Return a context manager that temporary sets `obj` attributes to the values specified in `new_attr_values`."""
    def set_attrs(attrs):
        for name, value in attrs.items():
            setattr(obj, name, value)

    saved_attr_values = {name: getattr(obj, name) for name in new_attr_values}
    try:
        set_attrs(new_attr_values)
        yield obj
    finally:
        set_attrs(saved_attr_values)


def assert_model_is(model, expected_model_type):
    actual_model_type = get_model_type(model)
    if actual_model_type != expected_model_type:
        raise TypeError(f"Expected {expected_model_type} but got {actual_model_type}.")
