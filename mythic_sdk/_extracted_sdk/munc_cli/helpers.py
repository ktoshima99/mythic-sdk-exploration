"""Helper functions useful for defining model conversion steps in application repos."""
from copy import copy
from functools import partial
import inspect
import re
import json
import logging
from pathlib import Path
import shutil
import tarfile
import tempfile
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import importlib
import os
import sys
import subprocess  # nosec B404

import coolname
from cloudpathlib import AnyPath, CloudPath
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import MissingMandatoryValue
from hydra.errors import CompactHydraException
from hydra.utils import instantiate
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core import plugins
from hydra.core.hydra_config import HydraConfig
from funcy import walk_values

from munc._onnx_model import ONNXModel
from munc._session import Session, DEBUG_DIR
from munc import _node_utils
from munc._torchnet import default_layer_factory
from munc.op_config import configure_ops
from munc._generate_artifact import generate_artifact
from munc._artifact.artifact_writer import Artifact


logger = logging.getLogger(__name__)


class SessionFromConfig(Session, AbstractContextManager):
    """A `munc.Session` context manager that automates loading and saving ONNX files.

    It loads an ONNX file, creates a `Session` for it based on a provided configuration, and
    when a user is done with the session, it saves the modified model to a file.

    Example
    -------
    config = {'src': 'data/mythic-model.onnx', 'dest': 'data/acm-model.onnx'}
    with SessionFromConfig(config) as sess:
        sess.convert_mythic_to_bcm(bcm_class_str='munc_digital')
    """

    def __init__(self, config, dataloader=None, model=None, allow_other_keys=False, save_model=True):
        """Create a session.

        Parameters
        ----------
        config : dict
            A configuration. Keys:
              src - a file to load an ONNX model from
              dest - a file to save a modified model to. Saving is disabled if this key is not specified.
              torchnet.layers - a list of dictionaries with keys 'pattern' and 'config'. It's used to create a torchnet
                                layer factory.
              All the other keys are passed to the Session init.
        dataloader : Dataloader, optional
            A data loader, it's passed to the session init.
        model : ONNXModel, optional
            An ONNX model. If specified it's passed to the session init, and the model specified in `src` is not used.
        allow_other_keys : bool, optional
            If true, unknown config keys are ignored, otherwise they cause an error.
        save_model : bool, optional
            If true, the modified model is saved to the file specified by `dest` after the session is done.
            Defaults to True.
        """
        self.config = config
        self.save_model = save_model
        assert 'hwconfig' not in config and 'noise_config' not in config, \
            ("'hwconfig' and 'noise_config' are not supported anymore. Hardware settings should now be passed to "
             "the specific conversion operations that require them.")
        if model is None:
            src_path = AnyPath(config['src'])
            logger.info(f'Loading ONNX model from {src_path}')
            if not src_path.exists():
                raise FileNotFoundError(f'No such file: {src_path}')

            model = ONNXModel(src_path)
        session_params = dict(config) if isinstance(config, dict) else OmegaConf.to_container(config, resolve=True)
        if 'torchnet' in session_params:
            torchnet_config = session_params['torchnet']
            if 'hw_model' in torchnet_config:
                hw_model_desc = ''
                if 'name' in torchnet_config['hw_model']:
                    hw_model_desc += f" Hardware model: {torchnet_config['hw_model']['name']};"
                if ('noise_config' in torchnet_config['hw_model']
                   and torchnet_config['hw_model']['noise_config']['name']):
                    hw_model_desc += f" Noise config: {torchnet_config['hw_model']['noise_config']['name']};"
                logger.info(f'Torchnet configuration: {hw_model_desc}')
            layer_factory = make_torchnet_layer_factory(torchnet_config['layers'])
            session_params['torchnet_layer_factory'] = layer_factory
            session_params['activation_ckpt_config'] = torchnet_config.get('activation_ckpt_config', None)

        for key in ['src', 'dest', 'torchnet']:
            session_params.pop(key, None)
        if allow_other_keys:
            known_args = set(inspect.getfullargspec(Session).args)
            for key in list(session_params.keys()):
                if key not in known_args:
                    session_params.pop(key, None)

        super().__init__(model, loader=dataloader, **session_params)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_model and exc_type is None and self.config.get('dest'):
            dest_path = AnyPath(self.config['dest'])
            logger.info(f'Saving ONNX model to {dest_path}')
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(dest_path)
        return False


def call_with_local_copy(func, input_file, output_file):
    """Call `func` with a local copy of `input_file` and copy its result to `output_file`.

    Both `input_file` and `output_file` can be either local files or cloud URLs. This is a helper function
    that makes easier to applying a function that works on local files to files stored in a cloud.

    Parameters
    ----------
    func : Callable[[Path], Path]
        A function that takes a local file, creates a new local file, and returns its path.
        It will be called with a local copy of `input_file` and a local file it returns will be copied to `output_file`.
    input_file : AnyPath
        An input file (can be a URL pointing to a remote file).
    output_file : AnyPath
        An output file (can be a URL pointing to a remote file).
    """
    input_file = AnyPath(input_file)
    local_input_file = Path(input_file.fspath) if isinstance(input_file, CloudPath) else input_file
    local_output_file = func(local_input_file)
    copy_file(local_output_file, output_file)


def copy_file(src, dest):
    """Copy a file between local or cloud locations.

    Parameters
    ----------
      src: Union[str, os.PathLike, CloudPath]
          A file to copy.
      dest: Union[str, os.PathLike, CloudPath]
          A target location.
    """
    input_file = AnyPath(src)
    output_file = AnyPath(dest)

    if not isinstance(output_file, CloudPath):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    if input_file != output_file:
        if not input_file.exists():
            raise FileNotFoundError(f'No such file: {input_file}')

        if isinstance(output_file, CloudPath):
            output_file.upload_from(input_file)
        elif isinstance(input_file, CloudPath):
            input_file.copy(output_file)
        else:
            shutil.copy(input_file, output_file)


def _load_onnx_from_checkpoint_tar(path):
    """Load an ONNX model from a tar checkpoint."""
    with tempfile.TemporaryDirectory() as temp_directory, tarfile.open(path, "r") as tar_file:
        temp_directory = Path(temp_directory)

        def file_with_ext(ext):
            return next(temp_directory.glob('*.' + ext))

        tar_file.extractall(temp_directory)
        model = ONNXModel(file_with_ext('onnx'))
        # optimizer_state_dict = torch.load(file_with_ext('pth'))

    return model


def convert_training_to_acm(config):
    """Load a training ONNX model from either an ONNX file or a tar checkpoint file, convert it to ACM, and save.

    Parameters
    ----------
    config: DictConfig
      Conversion parameters. See `SessionFromConfig`.
    """
    checkpoint_file = AnyPath(config['src'])

    if checkpoint_file.suffix == ".tar":
        if isinstance(checkpoint_file, CloudPath):
            # tarfile wants a local file.
            checkpoint_file = checkpoint_file.fspath
        model = _load_onnx_from_checkpoint_tar(checkpoint_file)
    else:
        model = ONNXModel(checkpoint_file)

    with SessionFromConfig(config, model=model) as sess:
        sess.convert_mythic_to_bcm(bcm_class_str='munc_fp')


def save_model_metrics(cfg, data):
    """Save metrics `data` to a json file specified in `cfg['metrics_file']`.

    Parameters
    ----------
    cfg : dict
        A configuration dictionary. The dictionary must have key `metrics_file`.
    data : dict
        Metrics data. The dictionary can containing arbitrary caller-provided data.
    """
    metrics_file = AnyPath(cfg['metrics_file'])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(data, f, indent=2)


def load_model_metrics(cfg):
    """Load metrics data from a json file specified in `cfg['metrics_file']`.

    Parameters
    ----------
    cfg: dict
        A configuration dictionary. The dictionary must have key `metrics_file`.

    Returns
    -------
    dict
        An empty dictionary is the specified files does not exist, otherwise metrics data loaded from the file
        by `json.load`.
    """
    metrics_file = AnyPath(cfg['metrics_file'])
    if metrics_file.exists():
        with metrics_file.open("r") as f:
            data = json.load(f)
    else:
        data = {}
    return data


def record_model_metrics(cfg, key, metrics):
    """Add `metrics` data with `key` to a json file specified in `cfg['metrics_file']`.

    Parameters
    ----------
    cfg: dict
        A configuration dictionary. The dictionary must have key `metrics_file`.
    key: str
        With which key store `metrics`.
    metrics: object
        Metrics data to add to the specified file.
    """
    data = load_model_metrics(cfg)
    data[key] = metrics
    save_model_metrics(cfg, data)


def format_metrics(per_hw_model_metrics, hw_model_re, metric_re, num_digits=3):
    """Format metrics for a table.

    Parameters
    ----------
    per_hw_model_metrics : dict
        A dictionary with keys being model names and values being dictionaries with metrics.
    hw_model_re : str
        A regular expression that matches model names. It's used to filter models. The first group of the match is used
        as a label in the table.
    metric_re : str
        A regular expression that matches metric names. It's used to filter metrics. The first group of the match is
        used as a label in the table.
    num_digits : int, optional
        The number of digits after the decimal point to display for metrics. Defaults to 3.

    Returns
    -------
    list
        A list of lists. The first list is a header, the rest are rows of the table.
    """
    if not per_hw_model_metrics:
        return []

    def label(name, regex):
        match = re.match(regex, name)
        if not match:
            return None
        elif match.groups():
            return match[1]
        else:
            return name

    # Only use results of a model if its name matches the hw_model_re. Replace names by labels.
    per_hw_model_metrics = [(label(k, hw_model_re), v) for k, v in per_hw_model_metrics.items()
                            if re.match(hw_model_re, k)]
    metric_names = list(per_hw_model_metrics[0][1].keys())
    # Only use a metric if its name matches the metric_re.
    metric_names = list(filter(lambda k: re.match(metric_re, k), metric_names))
    max_hw_model_name_len = max(map(lambda p: len(p[0]), per_hw_model_metrics))
    metric_labels = list(map(lambda s: label(s, metric_re), metric_names))
    metric_vals_per_model = [[f"{model_metrics[metric_name]:.{num_digits}f}" for metric_name in metric_names]
                             for _, model_metrics in per_hw_model_metrics]

    def max_value_width(i):
        return max(len(metric_vals[i]) for metric_vals in metric_vals_per_model)

    metric_field_widths = [max(len(metric_label), max_value_width(i)) for i, metric_label in enumerate(metric_labels)]
    headers = [" " * max_hw_model_name_len,
               *[f"{metric_label:>{width}}" for metric_label, width in zip(metric_labels, metric_field_widths)]]

    res = []
    res.append(headers)
    for (model, _), metric_vals in zip(per_hw_model_metrics, metric_vals_per_model):
        metric_vals = [f"{metric_val:>{field_width}}"
                       for metric_val, field_width in zip(metric_vals, metric_field_widths)]
        res.append([f'{model:{max_hw_model_name_len}}', *metric_vals])
    return res


def run_conversion_step(cfg, step, step_types):
    """Run a model conversion step.

    Parameters
    ----------
    cfg : DictConfig
        A configuration. `cfg[step]` is used as a step configuration. It will be passed to a function that implements
        the step. Each step has a type, which defaults to the step name and can be explicitly specified by the step
        configuration key `step_type`. To find a step implementation this function uses dictionary `step_types`,
        which maps step types names to corresponding step implementations. A step can take either one or two parameters.
        A configuration is passed as the first parameter and the step name as the second one.
    step : str
        The name of a step to execute.
    step_types: Dict[str, Union[Callable[[DictConfig], None], Callable[[DictConfig, str], None]]]
        Defined step types. A key is a step type name, a value is a function implementing the step.
    """
    step_config = cfg[step]
    OmegaConf.resolve(step_config)
    step_config = copy(step_config)
    with open_dict(step_config):
        step_type = step_config.pop('step_type', step)
    step_impl = step_types[step_type]
    num_params = len(inspect.signature(step_impl).parameters)
    if num_params == 1:
        step_impl(step_config)
    else:
        step_impl(step_config, step)


def _flatten_step_groups(items, step_groups, path=()):
    """Flatten a list of steps and step groups into a flat list of steps."""
    result = []
    for item in items:
        if item in step_groups:
            if item in path:
                raise ValueError(f"Cycle detected in step_groups: {' -> '.join(path + (item,))}")
            result.extend(_flatten_step_groups(step_groups[item], step_groups, path + (item,)))
        else:
            result.append(item)
    return result


def run_conversion_steps(cfg, step_types=None, run_init=True):
    """Run a model conversion steps specified in the configuration.

    Parameters
    ----------
    cfg : DictConfig
        A configuration. The following keys are used:
        step_order - a list of defined step in the order they should be executed.
        steps - a list of steps to run or a string to run a single step. Can also contain group names that
            reference step_groups. This value is used as a set, the execution order is defined by `step_order`.
        step_groups - optional dict mapping group names to lists of steps/groups.
        exclude_steps - a list of step to disable. Can also contain group names. A step specified in `steps` is
            only run if it is not in `exclude_steps`.
        A section for each step specified in `step_order`. It provides a step configuration and will be passed to
            a function implementing the step. See `run_conversion_step`.
    step_types: Optional[Dict[str, Callable[[DictConfig], None]]]
        Defined step types. A key is a step type name, a value is a function implementing the step. If not specified,
        step types will be taken from the configuration section `step_types` using `resolve_step_type_definitions`.
    run_init : bool, optional
        If true, a function provided as the step type `init` is run before any other steps. Defaults to True.
    """
    step_order = cfg['step_order']
    step_groups = cfg.get('step_groups', {})
    conflicting_names = set(step_groups.keys()) & set(step_order)
    if conflicting_names:
        raise ValueError(f"Names {conflicting_names} are used as both step groups and steps. "
                         "A name must be used exclusively as either a group or a step.")
    steps = cfg['steps']
    steps = [steps] if isinstance(steps, str) else steps
    steps = set(_flatten_step_groups(steps, step_groups))
    exclude_steps = set(_flatten_step_groups(cfg.get('exclude_steps', []), step_groups))
    enabled_steps = steps - exclude_steps
    if step_types is None:
        step_types = resolve_step_type_definitions(cfg['step_types'])

    def validate_steps(steps, collection, check_step_order_for_single_step):
        if check_step_order_for_single_step or len(steps) > 1:
            unknow_steps = steps - set(step_order)
            if unknow_steps:
                raise Exception(f"Unknown steps {unknow_steps} found in '{collection}'. Defined steps are {step_order}")
        for step in steps:
            if step not in cfg:
                raise Exception(f"Step {step} does not have a configuration.")
            step_type = cfg[step].get('step_type', step)
            if step_type not in step_types:
                raise Exception(f"Step {step} has unknow type {step_type}.")

    validate_steps(steps, 'steps', False)
    validate_steps(exclude_steps, 'exclude_steps', True)

    steps_to_run = list(filter(lambda s: s in enabled_steps, step_order)) if len(enabled_steps) > 1 else enabled_steps
    # Check for missing mandatory keys before running any steps.
    for step in steps_to_run:
        check_missing_hydra_config_keys(cfg[step])
    if run_init and 'init' in step_types:
        step_types['init'](cfg)
    for step in steps_to_run:
        run_conversion_step(cfg, step, step_types)


def get_integer_suffix(s, default=1):
    """Return an integer suffix of `s`.

    If `s` ends with a sequence of decimal digits, returns the int value of the suffix, otherwise returns `default`.
    """
    digits = re.search("[0-9]*$", s)[0]
    return int(digits) if digits else default


def make_sequential_file_name(parent_dir, get_index, make_name, glob='*', make_directory=False, min_index=1):
    """Make a unique file name that includes a sequential integer index.

    This function can be used to generate a sequence of uniquely numbered file names. The numbers are assigned in
    the file creation order. For example, `file1`, `file2`, `file3`, and so on.

    Parameters
    ----------
    parent_dir : Path
        A directory where to create a file.
    get_index : Callable[[Path], int]
        A function that takes a file and returns its number. It's used to find the smallest unused file number.
    make_name : Callable[[int], str]
        A function that creates a file name with given file number. It takes a file number and returns a file name.
    glob : str, optional
        A glob string that gets passed to `parent_dir.glob` to select files that are in the same "namespace" and
        should have unique numbers. Default to '*'.
    min_index : int, optional
        The minimum file index. It's used if there are no pre-existing numbered files in the directory. Defaults to 1.
    make_directory : bool, optional
        Create a directory instead of a file. Defaults to false.
    """
    while True:
        existing_files = parent_dir.glob(glob)
        next_index = max(map(get_index, existing_files), default=min_index - 1) + 1
        new_file = parent_dir / make_name(next_index)
        try:
            if make_directory:
                new_file.mkdir(parents=True)
            else:
                new_file.touch(exist_ok=False)
            return new_file
        except FileExistsError:
            pass


def create_run_dir(parent_dir):
    """Create a new directory with unique name within the parent directory.

    The directory name comprises two words generated by `coolname` and a unique integer suffix that is the largest
    integer file name suffix in the parent directory plus 1.
    """
    parent_dir = Path(parent_dir)
    temp_dir = make_sequential_file_name(parent_dir, make_directory=True,
                                         make_name=lambda n: f'file{n}',
                                         get_index=lambda path: get_integer_suffix(path.stem))
    num = temp_dir.name[4:]
    run_name = coolname.generate_slug(2) + ('' if num == '1' else '-' + num)
    run_dir = parent_dir / run_name
    temp_dir.rename(run_dir)
    return run_dir


def register_omega_config_now_resolver():
    """Register an OmegaConf resolver for the current date/time.

    `${now:DATETIME_FORMAT}` in a configuration file will be replaced by the current time formatted by `strftime`
     according to DATETIME_FORMAT. For example, `${now:%Y-%m-%dT%H:%M:%S%z}` -> `2023-09-01T00:14:30+0000`.
    """

    def timestamp_resolver(format):
        # Current time, timezone aware. Nice and simple :-)
        ts = datetime.now(timezone.utc).astimezone()
        return ts.strftime(format)

    OmegaConf.register_new_resolver("now", timestamp_resolver, replace=True)


def check_missing_hydra_config_keys(cfg):
    """Check if there are any missing mandatory Hydra configuration keys.

    Parameters
    ----------
    cfg : DictConfig
        A configuration.

    Raises
    ------
    CompactHydraException
        If there are missing mandatory keys.
    """
    try:
        OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    except MissingMandatoryValue as e:
        raise CompactHydraException(e)


def make_torchnet_layer_factory(configs, pattern_key='pattern', config_dict_key='config', priority_key='priority'):
    """Make the torchnet layer factory with a given configuration.

    Parameters
    ----------
    configs : Dict[str, Dict]
        Layer configurations. Keys are rule names (not used). Each configuration must have a pattern, a dict of layer
        parameters, and a priority. An entry with `pattern_key` is used to match a node using `_node_utils.match`
        function. An entry with `config_dict_key` is used to create a configuration dictionary for the node, i.e. it
        gets passed to a layer's constructor. An entry with `priority_key` determines the order in which rules are
        evaluated (lower values = higher priority).
    """
    def make_config_func(config):
        pattern = config[pattern_key]
        config_dict = walk_values(lambda v: instantiate(v) if isinstance(v, dict) else v, config[config_dict_key])

        def config_func(node):
            return config_dict if _node_utils.match(node, pattern) else {}

        return config_func

    # Sort configs by priority (lower values = higher priority)
    sorted_configs = sorted(configs.values(), key=lambda config: config[priority_key])
    config_funcs = [make_config_func(config) for config in sorted_configs]
    return partial(default_layer_factory, configs=config_funcs)


def configure_model_conversion_ops(ops, op_configs):
    """Configure Munc conversion `ops` using `op_configs`.

    Op configurations can be specified as a `ConfDict` or a regular Python dictionary.

    Parameters
    ----------
    ops : List[OpConf]
        A list of Munc ops to configure.
    op_configs : DictConfig | dict[str, dict[str, Any] | DictConfig]
        A dictionary mapping op names to their configurations.

    Returns
    -------
    List[OpConf]
        A list of op configurations with modified parameters.
    """
    # Convert DictConfig to dict and replace empty sections (None) with empty dicts.
    op_configs = OmegaConf.to_container(OmegaConf.create(op_configs), resolve=True)
    op_configs = walk_values(lambda x: {} if x is None else x, op_configs)
    return configure_ops(ops, op_configs)


def create_artifact_from_config(sess, config, artifact_conversion_ops=None):
    """Generate a Munc artifact.

    Parameters
    ----------
    sess : Session
        A Munc session to use.
    config : DictConfig
        A configuration. It must have the following keys:
            dest - a file to save the artifact to.
            artifact_directory - a directory where to store the artifact.
            keep_artifact_dir - if true, the artifact directory is not deleted after the artifact is created.
            include_debug - if true, debug information is included in the artifact.
            artifact - an artifact configuration. It will be passed to `generate_artifact` as keyword arguments.
            conversion_parameters.ops - a configuration for model conversion ops. See `configure_model_conversion_ops`.

    artifact_conversion_ops : List[OpConf], optional
        A list of artifact conversion operations. If not specified, the default list is
        used (`Sesson.get_bcm_to_artifact_conversion_ops`).
    """
    source_model = sess.model.deepcopy()
    local_artifact_file = Path(config.artifact_directory) / 'compiler_ready_artifact.tar.gz'
    config.artifact.debug_dir = (config.artifact.debug_dir or DEBUG_DIR) if config.include_debug else None
    artifact_config_dict = OmegaConf.to_container(config.artifact, resolve=True)
    artifact_conversion_ops = artifact_conversion_ops or sess.get_bcm_to_artifact_conversion_ops()
    sess.run_ops(*configure_model_conversion_ops(artifact_conversion_ops, config.conversion_parameters.ops))

    with Artifact(out_file=local_artifact_file, artifact_root_dir=config.artifact_directory,
                  keep_artifact_dir=config.keep_artifact_dir) as a:
        generate_artifact(sess, artifact_dir=a.artifact_dir, directory_contents=a.metadata, source_model=source_model,
                          **artifact_config_dict)
    logger.info(f'Saving the artifact to {config.dest}...')
    copy_file(local_artifact_file, config.dest)


def add_image_shape_to_compiler_config_file_name(config):
    """Automatically select a compiler config based on the input shapes.

    `{IMGSZ_input_name}` in the compiler config file name will be replaced with the actual input shape.
    """
    # It seems reasonable to do this compiler config name formatting in `compile_artifact` after
    # the artifact is unpacked, but `compile_artifact` takes a config, not its name. There are reasons for that:
    # 1. We may not want `compile_artifact` to know about the external configuration format.
    # 2. It may make sense to migrate compiler configs to Hydra, then they will not be separate files at all.
    compiler_config_name = config.compiler_config
    if "{IMGSZ_" in compiler_config_name:
        input_file = AnyPath(config.src)
        local_input_file = Path(input_file.fspath) if isinstance(input_file, CloudPath) else input_file
        config.src = str(local_input_file)
        with Artifact(in_file=config.src, keep_artifact_dir=False) as a:
            for input_name, input_shape in a.metadata["input_shapes"].items():
                input_shape_str = "x".join(map(str, reversed(input_shape)))
                compiler_config_name = compiler_config_name.replace(f"{{IMGSZ_{input_name}}}", input_shape_str)
            config.compiler_config = compiler_config_name


def compile_munc_artifact(config):
    """Generate firmware from a Munc artifact."""
    # We don't do these imports at the top, because the dependency on mythic-compiler is optional.
    logger.info(f'Compiling Munc artifact {config.src}...')
    add_image_shape_to_compiler_config_file_name(config)
    mythic_root_modified = False
    try:
        in_docker_container = Path("/.dockerenv").exists()
        if in_docker_container and 'MYTHIC_ROOT' not in os.environ:
            # Compilation requires MYTHIC_ROOT set to a path that is mapped one-to-one between a host and
            # its Docker containers for the Docker-in-Docker trick to work.
            # Typically it's /data/local/something.
            os.environ['MYTHIC_ROOT'] = config.mythic_root
            mythic_root_modified = True
        config_path = Path(config.compiler_config).absolute()
        subprocess.run(["mythic-compiler", "-cn", config_path.name, "-cp", str(config_path.parent),  # nosec B607 B603
                        f"src={config.src}", f"dest={config.dest}"],
                       check=True)
    finally:
        if mythic_root_modified:
            del os.environ['MYTHIC_ROOT']


def resolve_function(func_path, kind):
    """Resolve a fully qualified function name to a callable object.

    Parameters
    ----------
    func_path : str
        A fully qualified function name, e.g. `module.submodule.func_name`.
    kind : str
        A kind of the function being resolved, used in error messages.

    Returns
    -------
    Callable
        A callable object representing the function.
    """
    try:
        last_dot = func_path.rindex('.')
    except ValueError:
        raise ValueError(f"Invalid {kind} '{func_path}'. It must be a fully qualified function name.")
    module = func_path[:last_dot]
    func_name = func_path[last_dot + 1:]
    try:
        module = importlib.import_module(module)
    except ModuleNotFoundError:
        raise ValueError(f"Invalid {kind} '{func_path}'. Module '{module}' not found.")
    if not hasattr(module, func_name):
        raise ValueError(f"Invalid {kind} '{func_path}'. Function '{func_name}' not found.")
    return getattr(module, func_name)


def resolve_step_type_definitions(defs):
    """Convert a step definition configuration `{'step_name': 'module.func_name', ...}` to `{'step_name': func, ...}`.

    Parameters
    ----------
    defs : Dict[str, str]
        A dictionary where keys are step type names and values are fully qualified function names.

    Returns
    -------
    Dict[str, Callable]
        A dictionary where keys are step type names and values are functions implementing the steps.
    """
    return {name: resolve_function(func_path, 'step type definition') for name, func_path in defs.items()}


def symlink_hydra_output_dir(dest):
    """Create a symbolic link from Hydra's output directory to `dest`."""
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except ValueError:
        output_dir = None

    if output_dir is not None:
        Path(dest).symlink_to(Path(output_dir))


def add_unhandled_exception_logger():
    """Set up a global exception handler that logs unhandled exceptions via Python's `logging`.

    This function configures a custom exception hook that logs any unhandled exceptions
    using the standard logging framework.
    """
    prev_hook = sys.excepthook

    def _handle_exception(exc_type, exc_value, exc_traceback):
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        if prev_hook:
            prev_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _handle_exception

    # Disable Hydra's error handler, because it captures unhandled exceptions, prints it, and calls `exit(1)`, so
    # sys.excepthook does not get called.
    os.environ["HYDRA_FULL_ERROR"] = "1"


class HydraSearchPathPlugin(SearchPathPlugin):
    """A Hydra search path plugin to make Munc configs available to Munc users."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Add Munc config paths to the Hydra search path."""
        search_path.append(provider="mythic-munc", path="pkg://munc.hydra_configs")


def add_munc_configs_to_hydra_search_path():
    """Register the Hydra search path plugin for common model conversion configs."""
    plugins.Plugins.instance().register(HydraSearchPathPlugin)
