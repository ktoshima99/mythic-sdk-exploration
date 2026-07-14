# Copyright (C) 2024, Mythic Inc. All rights reserved.
#
#
"""The default implementations of some model conversion steps."""

import logging
import math
import os
import subprocess
import sys

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from cloudpathlib import AnyPath

from munc import ONNXType, ops
from munc.cli.helpers import (copy_file, load_model_metrics, format_metrics, configure_model_conversion_ops,
                              SessionFromConfig, create_artifact_from_config, record_model_metrics,
                              resolve_function)
from munc.op_config import op_conf_seq, op_do
from munc._session_tools import get_model_type
from munc.cli.monte_carlo import collect_accuracy_data, load_accuracy_data, process_accuracy_data, get_num_samples


logger = logging.getLogger(__name__)


try:
    import wandb
except ImportError:
    wandb = None


def get_rank():
    """Return the Distributed Data Parallel (DDP) rank of this process.

    Returns
    -------
    int
        The rank of this process in a distributed training setup. Returns -1 if not
        running in a distributed environment.
    """
    return int(os.getenv('RANK', -1))


def wandb_init(wandb_init_args, config, force=False):
    """Initialize Weights & Biases (W&B) logging if available and not already initialized.

    This function safely initializes W&B logging, handling cases where W&B is not installed
    or already initialized. It's designed to be called from model conversion steps that
    want to log metrics and configuration to W&B.

    Parameters
    ----------
    wandb_init_args : dict
        Parameters to be passed to `wandb.init()`. Common keys include 'project', 'name',
        'tags', etc.
    config : dict
        A configuration dictionary to log to W&B. This typically contains the full
        configuration used for the conversion step.
    force : bool, optional
        If False and W&B is already initialized, this function will do nothing.
        If True, W&B will be initialized regardless. Defaults to False.
    """
    if wandb and hasattr(wandb, 'run') and (force or not wandb.run):
        wandb.init(config=config, **wandb_init_args)


def copy_artifact_step(config):
    """Copy an artifact file from source to destination location.

    This step is typically used to copy artifacts from local storage to cloud storage
    or between different storage locations. It handles both local and cloud paths.

    Parameters
    ----------
    config : DictConfig
        Configuration containing:
        - src : str or Path
            Source path of the artifact to copy
        - dest : str or Path
            Destination path where the artifact should be copied
    """
    logger.info(f'Copying an artifact from {config.src} to {config.dest}')
    copy_file(config.src, config.dest)


def summarize_metrics_step(cfg):
    """Format and log model evaluation metrics in a tabular format.

    This step loads metrics from a JSON file, formats them into a readable table
    using regular expressions to filter models and metrics, and logs the results.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing:
        - metrics_file : str or Path
            Path to the JSON file containing metrics data
        - model_re : str
            Regular expression to match and filter model names
        - metric_re : str
            Regular expression to match and filter metric names
        - num_digits : int, optional
            Number of digits after decimal point. Defaults to 3.
    """
    for row in format_metrics(load_model_metrics(cfg), cfg.model_re, cfg.metric_re, cfg.get("num_digits", 3)):
        logger.info("  ".join(row))


def delete_auxiliary_outputs(model, num_outputs_to_keep=1, delete_from_beginning=False):
    """Delete auxiliary model outputs, keeping only the specified number of main outputs.

    This function is typically used to clean up models that have additional debugging
    or intermediate outputs that are not needed for deployment. After calling this
    function, you should call `ops.RemoveDanglingNodes()` to remove any nodes that
    are no longer connected to outputs.

    Parameters
    ----------
    model : ONNXModel
        The ONNX model to modify in-place.
    num_outputs_to_keep : int, optional
        Number of outputs to retain in the model. Defaults to 1.
    delete_from_beginning : bool, optional
        If True, delete outputs starting from index 0. If False, delete outputs
        from the end. Defaults to False.
    """
    while len(model.get_output_names()) > num_outputs_to_keep:
        if delete_from_beginning:
            model.pop_output(0)
        else:
            model.pop_output()


def run_conversion(config: DictConfig, convert):
    """Create a Munc session and execute a conversion function with real data statistics.

    This helper function sets up a Munc session with a model and dataloader as specified
    in the configuration, then calls the provided conversion function. It's designed for
    conversion steps that need to collect statistics from real data (e.g., activation
    ranges for quantization).

    Parameters
    ----------
    config : DictConfig
        Configuration containing:
        - src : str or Path
            Path to the input model file
        - dest : str or Path
            Path where the converted model should be saved
        - dataloader : dict
            Dataloader configuration with 'factory' key specifying the fully
            qualified function name to create the dataloader
        - Additional keys for session configuration
    convert : Callable[[Session], Any]
        A function that takes a Munc Session and performs the conversion operations.

    Returns
    -------
    Any
        The return value of the convert function.
    """
    if 'dataloader' in config:
        dataloader_config = OmegaConf.create(config.dataloader, flags={"struct": False})
        dataloader_factory = dataloader_config.pop('factory')
        make_dataloader = resolve_function(dataloader_factory, 'data loader factory')
        dataloader = make_dataloader(dataloader_config)
    else:
        dataloader = None
    with SessionFromConfig(config, dataloader, allow_other_keys=True) as sess:
        return convert(sess)


def to_training_step(config):
    """Convert a floating-point ONNX model to a Mythic Node training model.

    This step transforms a standard floating-point ONNX model into a Mythic Node model
    that can be used for hardware-aware training. The conversion
    includes inserting Mythic-specific nodes and configuring them based on the
    specified hardware model.

    Parameters
    ----------
    config : DictConfig
        Configuration containing:
        - src : str or Path
            Path to the input floating-point ONNX model
        - dest : str or Path
            Path where the converted training model should be saved
        - torchnet.hw_model.hardware_config_name : str
            Name of the hardware configuration to use for conversion
        - conversion_parameters.options : dict
            Additional options for the conversion operations
        - conversion_parameters.ops : dict
            Configuration for individual conversion operations
        - dataloader : dict
            Dataloader configuration for collecting statistics
    """
    hardware_config_name = config.torchnet.hw_model.hardware_config_name

    def convert(sess):
        ops = sess.get_original_to_mythic_conversion_ops(hardware_config_name=hardware_config_name,
                                                         **config.conversion_parameters.options)
        sess.run_ops(*configure_model_conversion_ops(ops, config.conversion_parameters.ops))

    run_conversion(config, convert)


def convert_training_to_acm(config: DictConfig,
                            extra_ops=(),
                            extra_ops_after_conversion=(),
                            op_configs_override={},
                            ):
    """Convert a Mythic Node training model to an ACM (Analog Compute Model).

    This helper function transforms a Mythic Node training model into an ACM model.

    Parameters
    ----------
    config : DictConfig
        Configuration containing:
        - src : str or Path
            Path to the input Mythic Node training model
        - dest : str or Path
            Path where the converted ACM model should be saved
        - dataloader : dict
            Dataloader configuration for collecting statistics
    extra_ops : sequence, optional
        Additional operations to run before the standard conversion operations.
    extra_ops_after_conversion : sequence, optional
        Additional operations to run after the standard conversion operations.
    op_configs_override : dict, optional
        Configuration overrides for specific operations. Keys are operation names,
        values are dictionaries of configuration parameters.
    """
    def convert(sess):
        def is_first_on_chip(node):
            return (node.on_chip
                    and any(n and not n.on_chip for n in map(sess.model.get_node_with_output_name, node.input)))

        # If all the off-chip to on-chip go to Mythic Convs with 3 input channels, we can pad to 3 instead of 8.
        first_layer_can_be_padded_to_3 = all(n.op_type == ONNXType.MYTHIC_CONV and n.initializer[1].shape[1] == 3
                                             for n in sess.model.get_nodes_with_filter(is_first_on_chip))
        op_configs = {"ConvertConvsToBCM": dict(bcm_class_str="munc_fp"),
                      "ChannelPaddingTo8": dict(number_of_input_channels=3 if first_layer_can_be_padded_to_3 else 8)}
        op_configs.update(op_configs_override)
        op_configs.update(config.get("conversion_parameters", {}).get("ops", {}))
        op_list = list(extra_ops) + sess.get_mythic_to_bcm_conversion_ops() + list(extra_ops_after_conversion)
        sess.run_ops(*configure_model_conversion_ops(op_list, op_configs))
    run_conversion(config, convert)


def convert_training_to_acm_step(config: DictConfig):
    """Convert a Mythic Node training model to ACM (Analog Compute Model).

    This step transforms a Mythic Node training model into an Analog Compute Model (ACM).
    It's a wrapper around `convert_training_to_acm` that can be used directly as a conversion step.

    Parameters
    ----------
    config : DictConfig
        Configuration containing:
        - src : str or Path
            Path to the input Mythic Node training model
        - dest : str or Path
            Path where the converted ACM model should be saved
        - dataloader : dict
            Dataloader configuration for collecting statistics
    """
    convert_training_to_acm(config)


def create_training_artifact(config, extra_ops=(), num_outputs_to_keep=1000,
                             delete_outputs_from_beginning=False):
    """Generate a Munc artifact from an ACM model.

    This helper function creates a compressed artifact file that can be used as a Mythic compiler input.
    It containing the model and metadata needed for compilation.

    Parameters
    ----------
    config : DictConfig
        Configuration containing:
        - src : str or Path
            Path to the input ACM model
        - dest : str or Path
            Path where the artifact should be saved
        - artifact_directory : str or Path
            Directory for temporary artifact files
        - include_debug : bool
            Whether to include debug information in the artifact
        - keep_artifact_dir : bool
            Whether to keep the temporary artifact directory after creation
        - dataloader : dict
            Dataloader configuration for collecting statistics
    extra_ops : tuple, optional
        Additional operations to run before creating the artifact.
    num_outputs_to_keep : int, optional
        Maximum number of model outputs to retain. Defaults to 1000.
    delete_outputs_from_beginning : bool, optional
        If True, delete outputs starting from index 0. If False, delete from the end.
        Defaults to False.
    """
    def helper(sess):
        delete_auxiliary_outputs_ops = op_conf_seq(
            op_do(lambda op: delete_auxiliary_outputs(op.model, num_outputs_to_keep,
                                                      delete_from_beginning=delete_outputs_from_beginning),
                  pass_op=True),
            ops.RemoveDanglingNodes,
        )
        artifact_conversion_ops = op_conf_seq(*delete_auxiliary_outputs_ops, *extra_ops,
                                              *sess.get_bcm_to_artifact_conversion_ops())
        create_artifact_from_config(sess, config, artifact_conversion_ops=artifact_conversion_ops)

    # SessionFromConfig uses `dest` as an ONNX output file. That's not what we want here.
    sess_config = config.copy()
    sess_config.dest = None
    run_conversion(sess_config, helper)


def create_training_artifact_step(config):
    """Generate a Munc deployment artifact from an ACM model.

    This function creates a compressed artifact file that can be used as a Mythic compiler input.
    It's a wrapper around `create_training_artifact` that can be used directly as a conversion step.

    Parameters
    ----------
    config : DictConfig
        Configuration containing:
        - src : str or Path
            Path to the input ACM model
        - dest : str or Path
            Path where the artifact should be saved
        - artifact_directory : str or Path
            Directory for temporary artifact files
        - include_debug : bool
            Whether to include debug information in the artifact
        - keep_artifact_dir : bool
            Whether to keep the temporary artifact directory after creation
        - dataloader : dict
            Dataloader configuration for collecting statistics
    """
    create_training_artifact(config)


def run_evaluator(step_config, session):
    """Evaluate an ONNX model using a specified evaluator function.

    Parameters
    ----------
    step_config : dict or DictConfig
        Configuration containing evaluator settings. Must have an 'evaluator_config' key
        with an 'evaluator' field specifying the fully qualified function name of the
        evaluator to use.
    session : Session
        A Munc session containing the model to evaluate.

    Returns
    -------
    object
        The result returned by the evaluator function. The type depends on the specific
        evaluator used.
    """
    step_config = OmegaConf.create(step_config, flags={"struct": False})
    evaluator = step_config.evaluator_config.pop('evaluator')
    evaluator_func = resolve_function(evaluator, "evaluator function")
    return evaluator_func(step_config, session)


def eval_acm_step(cfg):
    """Evaluate an ACM model using a specified hardware configuration and store metrics.

    This step loads an ACM model, switches it to the specified hardware model configuration,
    runs evaluation using the configured evaluator function, and records the resulting
    metrics to a JSON file for later analysis.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing:
        - src : str or Path
            Path to the ACM model file to evaluate
        - metrics_file : str or Path
            Path where metrics will be stored
        - acm_model : str
            Name of the ACM model type (e.g., 'munc_fp', 'munc_digital', 'munc_acm_signoff')
        - acm_model_config : str, optional
            Specific configuration variant for the ACM model (e.g., 'v0p4', 'v0p5')
        - evaluator_config : dict
            Configuration for the evaluator, must contain 'evaluator' key with
            the fully qualified function name of the evaluator to use
        - Additional keys may be present for session configuration
    """
    with SessionFromConfig(cfg, allow_other_keys=True) as s:
        s.run_ops(ops.SwitchBCM(bcm_class_str=cfg.acm_model, bcm_attr_str=cfg.acm_model_config))
        metrics = run_evaluator(cfg, s)
        config_suffix = ':' + cfg.acm_model_config if cfg.acm_model_config else ''
        record_model_metrics(cfg, cfg.get('model_type', cfg.acm_model + config_suffix), metrics)


def predict_with_acm_step(cfg):
    """Run inference on an ACM model using a specified hardware configuration.

    This step loads an ACM model, switches it to the specified hardware model configuration,
    and runs inference (prediction) using the configured evaluator function. Unlike
    evaluation steps, this is typically used for generating predictions on new data
    rather than computing accuracy metrics.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing:
        - src : str or Path
            Path to the ACM model file to use for prediction
        - acm_model : str
            Name of the ACM model type (e.g., 'munc_fp', 'munc_digital', 'munc_acm_signoff')
        - acm_model_config : str, optional
            Specific configuration variant for the ACM model (e.g., 'v0p4', 'v0p5')
        - evaluator_config : dict
            Configuration for the evaluator, must contain 'evaluator' key with
            the fully qualified function name of the evaluator to use
        - Additional keys may be present for session configuration
    """
    with SessionFromConfig(cfg, allow_other_keys=True) as s:
        s.run_ops(ops.SwitchBCM(bcm_class_str=cfg.acm_model, bcm_attr_str=cfg.acm_model_config))
        with open_dict(cfg):
            cfg.evaluator_config.evaluator_mode = 'predict'
        run_evaluator(cfg, s)


def eval_onnx_step(cfg):
    """Evaluate an ONNX model and store computed metrics.

    Creates a session from the configuration, runs the specified evaluator function,
    and records the resulting metrics using `record_model_metrics`. The model must
    be compatible with TorchNet.

    Parameters
    ----------
    cfg : dict or DictConfig
        Configuration containing:
        - src : str or Path
            Path to the ONNX model file to evaluate
        - metrics_file : str or Path
            Path where metrics will be stored
        - evaluator_config : dict
            Configuration for the evaluator, must contain 'evaluator' key with
            the fully qualified function name of the evaluator to use
        - Additional keys may be present for session configuration
    """
    with SessionFromConfig(cfg, allow_other_keys=True) as s:
        metrics = run_evaluator(cfg, s)
        record_model_metrics(cfg, cfg.get('model_type', str(get_model_type(s.model))), metrics)


def get_dummy_torchnet_dataloader(dataloader_config: DictConfig):
    """Take a dataloader configuration and return None.

    This function can be used as a conversion dataloader factory to instruct Munc to use a dummy dataloader that
    produces random data. It can be used as a value of `conversion_dataloader.factory`.
    """
    return None


def collect_accuracy_data_sequential(cfg, start_index=0):
    """Run Monte Carlo accuracy collection in a single process."""
    with SessionFromConfig(cfg, allow_other_keys=True, save_model=False) as sess:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        collect_accuracy_data(cfg_dict, cfg.dest, sess, run_evaluator, start_index)


def _get_available_cuda_devices():
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    return ([dev.strip() for dev in visible_devices.split(',') if dev.strip()] if visible_devices
            else [str(idx) for idx in range(torch.cuda.device_count())])


def collect_accuracy_data_parallel(cfg, num_proc, step_name):
    """Shard Monte Carlo accuracy collection across multiple GPUs via subprocesses."""
    num_samples = get_num_samples(cfg)
    # Parallelize over the first schedule step only for now to avoid sharing partial hardware configurations across
    # processes.
    min_chunk_size = get_num_samples(cfg, 1)
    num_chunks = math.ceil(num_samples / min_chunk_size)
    available_devices = _get_available_cuda_devices()
    active_procs = min(num_proc, num_chunks, len(available_devices))
    chunks_per_gpu, remainder = divmod(num_chunks, active_procs)

    processes = []
    env = os.environ.copy()
    try:
        keys_to_strip = ["steps"] + [f"{step_name}.{key}" for key in ["nproc", "num_samples"]]
        base_args = list(filter(lambda p: all(not p.startswith(f'{key}=') for key in keys_to_strip), sys.argv))
        args = base_args + [f'{step_name}.nproc=1', f'steps={step_name}']
        proc_first_file_num = 0
        for gpu_idx, cuda_device in enumerate(available_devices[:active_procs]):
            chunks_for_proc = chunks_per_gpu + (1 if gpu_idx < remainder else 0)
            samples_for_proc = chunks_for_proc * min_chunk_size
            env['CUDA_VISIBLE_DEVICES'] = cuda_device
            logger.info(f'Starting {step_name} on GPU {cuda_device} (index {gpu_idx})')
            cmd = [sys.executable, *args, f'{step_name}.num_samples={samples_for_proc}',
                   f'++{step_name}.start_index={proc_first_file_num}']
            processes.append(subprocess.Popen(cmd, env=env))
            proc_first_file_num += samples_for_proc

        for proc in processes:
            retcode = proc.wait()
            if retcode:
                raise RuntimeError(f'{step_name} process exited with status {retcode}')
    finally:
        # Ensure no child processes keep running on errors or interrupts.
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


def collect_accuracy_data_step(cfg, step_name):
    """Run Monte Carlo accuracy evaluation and persist raw metrics for later aggregation.

    This step builds a Munc session from ``cfg``, repeatedly randomizes the hardware
    parameters according to ``cfg.schedule``/``cfg.num_samples``, and records the
    evaluator output for each sample under ``cfg.dest`` as ``metrics_XXXX.json``.
    The destination directory must not already exist; it is created before any work begins to prevent mixing results
    from different runs.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing:
        - dest : str or Path
            Output directory where per-sample metric JSON files are written.
        - evaluator_config / dataloader / model parameters
            Passed through to ``SessionFromConfig`` and ultimately to
            ``collect_accuracy_data`` and ``run_evaluator``.
        - num_samples or schedule : dict
            Monte Carlo sampling settings used to determine how many randomized
            sessions to evaluate.
        - nproc : int, optional
            Number of GPUs / worker processes to use; defaults to
            ``torch.cuda.device_count()``.
    step_name : str
        Pipeline step identifier, used when respawning the step in parallel mode.

    Notes
    -----
    If ``nproc`` is greater than one, work is sharded across GPUs by spawning
    subprocesses, each evaluating an integral number of schedule chunks on its
    assigned device. Otherwise, evaluation runs serially within a single process.
    """
    output_dir = AnyPath(cfg.dest)
    # Refuse to write results to an existing directory for now to avoid mixing different runs.
    output_dir.mkdir(parents=True, exist_ok=('start_index' in cfg))

    num_available_gpus = max(1, torch.cuda.device_count())
    # Use all the GPUs if 'nproc' is set to empty string or None.
    num_proc = cfg.get('nproc') or num_available_gpus
    if num_proc > num_available_gpus:
        logger.warn(f'Requested {num_proc} processes but only {num_available_gpus} GPUs are available')
        num_proc = num_available_gpus

    if num_proc == 1:
        collect_accuracy_data_sequential(cfg, cfg.get('start_index', 0))
    else:
        collect_accuracy_data_parallel(cfg, num_proc, step_name)


def process_accuracy_data_step(cfg):
    """Aggregate Monte Carlo accuracy samples into confidence bounds and log them.

    Reads the per-sample metrics produced by ``collect_accuracy_data_step`` from
    ``cfg.src`` (``metrics_*.json`` files), extracts values for ``cfg.model_type``,
    and computes lower tolerance bounds via ``process_accuracy_data`` using the
    desired coverage ``cfg.prop`` and confidence level ``cfg.confidence``. The
    resulting summary metrics are written with ``record_model_metrics`` under
    ``cfg.model_type`` and then formatted for logging by ``summarize_metrics_step``.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing:
        - src : str or Path
            Directory holding the raw Monte Carlo metric JSON files.
        - model_type : str
            Key inside each metrics file to read (e.g., ONNX model type).
        - prop : float
            Desired proportion of the metric distribution to cover.
        - confidence : float
            Confidence level for the lower tolerance bound computation.
        - metric_keys : list[str], optional
            Subset of metrics to process; defaults to all metrics present.
        - model_re / metric_re : str, optional
            Regex filters forwarded to ``summarize_metrics_step`` for display.
    """
    metrics = load_accuracy_data(cfg.src, cfg.model_type)
    processed_metrics = process_accuracy_data(metrics, cfg.prop, cfg.confidence, metric_keys=cfg.get('metric_keys'),
                                              include_mean_std=True)
    record_model_metrics(cfg, cfg.model_type, processed_metrics)
    summarize_metrics_step(cfg)


def optimize_ifsr_step(config: DictConfig):
    """Run the OptimizeIFSR op."""
    run_conversion(config, lambda sess: sess.run_ops(ops.OptimizeIFSR(force_per_channel=True, decrease_only=True)))
