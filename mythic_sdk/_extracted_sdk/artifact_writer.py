"""Create inference artifacts."""
from contextlib import AbstractContextManager
import logging
import os
import shutil
import json
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import onnx
import tensorflow as tf
import tf_keras
from ai_edge_litert.interpreter import Interpreter
from funcy import project
from omegaconf import OmegaConf

from munc._artifact._prepare_off_on_chip_transitions import insert_type_and_layout_conversions

from munc._session import Session
from munc import ops
from munc._constants import ONNXType


logger = logging.getLogger(__name__)


MUNC_ARTIFACT_DIRECTORY_NAME = 'compiler_ready_artifact'


def create_offchip_graph(name, models, data_formats):
    """Add a transformed ONNX graph that are ready for for silicon inference."""
    models = dict(models)
    model = models[name]
    model = model.deepcopy()
    models[name] = model
    insert_type_and_layout_conversions(models, name, *data_formats)
    return model


def _modify_key_for_h5_hierarchy(key, input_name):
    """Modify key if form is <str_base>_nnnnn to nnnn/<str_base>."""
    if key[-5:].isdigit() and key[-6:-5] == '_':
        # Separate base and id
        key_base = key[:-6]
        sample_id = key[-5:]

        # 'data' is special here because it also appears in the key used by the activation data whose key is
        # 'nnnnn/data/edge_name'. So having 'nnnn/data' will result in name collision with
        # group name. Use nnnnn/data/input_name to store the input data from the data generator.
        if key_base == "data":
            key = f"{sample_id}/data/{input_name}"
        else:
            key = f"{sample_id}/{key_base}"
    return key


def _write_data_to_h5_file(artifact, filename, input_name):
    """Write artifact data in h5 format.

    Expected artifact dictionary keys:
    - data: {}
    - types: {}
    - labels: {label_00000: <some_struct>, ..., label_00006: <some_struct>}
    - paths: {path_00000: <path_str>, ..., path_00005: <path_str>}
    - shapes: {shape_00000: <shape_struct>, ..., shape_00005: <shape_struct>}
    - 00000/data/<first_node_name>: <np.array>
    - ...
    - <nsamples>/data/<last_node_name>: <np.array>
    - <nsamples> = '00005' if there are 5 samples.

    Store data in groups identified by the sample numbers converted to strings, e.g. data from the 50th sample are
    saved under a group titled: '00050'. Non-array and non-regular-array data are converted into strings and stored as
    a group attribute. Data corresponding to a specific edge in the graph, including the input images, are stored in
    the subgroup 'data' under the name of the edge.

    When transformed into h5, the data will be laid out as follows:

    ```
    00000: {label: <label_struct_for_sample_00000, data: {}}
        00000.attrs: {path: <path of 00000>, shape: <shape of 00000>}
        00000/data: {first_node_name: <array>, ..., list_node_name: <array>}

    00001:
    ...
    ```
    """
    hdf5_use_file_locking_orig = os.environ.get("HDF5_USE_FILE_LOCKING", '')
    try:
        os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"
        f = h5py.File(filename, "w")
        for dset_name, dset_value in artifact.items():
            if not isinstance(dset_value, dict):
                if isinstance(dset_value, np.ndarray):
                    f.create_dataset(dset_name, data=dset_value)
                continue

            for _key, _val in dset_value.items():
                if isinstance(_val, (tuple, list, str)):
                    # Will store as  string as an attribute of the group
                    # Split out key's base and sample number that's used as group name
                    _key_mod, grp_name = _key[:-6], _key[-5:]

                    # Convert to string if needed
                    f[grp_name].attrs[_key_mod] = _val if isinstance(_val, str) else json.dumps(_val)
                else:
                    _key_mod = _modify_key_for_h5_hierarchy(_key, input_name)
                    f.create_dataset(_key_mod, data=_val)
    finally:
        os.environ['HDF5_USE_FILE_LOCKING'] = hdf5_use_file_locking_orig


def _get_model_input_name(model_input_names):
    """Return the only elemment of `model_input_names` if it's a single element list, otherwise 'sample_input'."""
    return (model_input_names[0] if isinstance(model_input_names, list) and len(model_input_names) == 1
            else 'sample_input')


def get_reference_dir(model_dir):
    """Return the artifact's reference directory."""
    return model_dir / 'reference'


def model_relative_path(path, model_dir):
    """Return the relative path of `path` with respect to `model_dir`."""
    return str(path.relative_to(model_dir))


def write_artifact(model_dir, subgraphs_bcm,
                   op_type_count,
                   source_model=None,
                   expected_input='RGB',
                   padding_value=0, input_shapes=None,
                   debug_dir=None):
    """Write a Munc artifact`.

    Creates a new directory called `model_dir`, and writes the _artifact components into it, prepending
    `MUNC_ARTIFACT_DIRECTORY_NAME` to each component.

    Parameters
    ----------
    model_dir : str
        Path-like string pointing towards directory in which the artifact components will be saved.
    subgraphs_bcm : Dict[str, ONNXModel]
        Models subgraphs. Keys are section names, values are corresponding ONNX models.
    source_model : Optional[ONNXModel]
        A source model this artifact is based on. If specified it will be included into the artifact.
    op_type_count : Dict
        Op counts.
    expected_input : str, optional
        Expected input format. Defaults to 'RGB'.
    padding_value : int, optional
        Padding value for the input data. Defaults to 0.
    input_shapes : dict, optional
        A shape for each input. Defaults to None.
    debug_dir : Optional[PathLike]
        A directory with debug information. If specified, it will be copied into the model directory.

    Returns
    -------
    Dict
        Dictionary with artifact metadata (the paths to the written files, etc).
    """
    model_dir = Path(model_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    reference_dir = get_reference_dir(model_dir)
    reference_dir.mkdir()
    if debug_dir and debug_dir.is_dir():
        shutil.copytree(debug_dir, model_dir / "debug")

    def relative_path(path):
        return model_relative_path(path, model_dir)

    onnx_graphs = {}
    for name, model in subgraphs_bcm.items():
        bcm = "_bcm" if "on_chip" in name else ""
        filename = reference_dir / f"{MUNC_ARTIFACT_DIRECTORY_NAME}_{name}{bcm}.onnx"
        model.save(filename)
        onnx_graphs[name] = relative_path(filename)

    if source_model:
        dst_filename_retrained = reference_dir / 'source_model.onnx'
        source_model.save(dst_filename_retrained)
        onnx_graphs['retrained_model_filename'] = relative_path(dst_filename_retrained)

    with open(model_dir / 'op_type_count.json', "w") as f:
        json.dump(op_type_count, f)

    directory_contents = {
        'onnx_graphs': onnx_graphs,
        'expected_input': expected_input,
        'padding_value': padding_value,
        'input_shapes': input_shapes,
    }

    return directory_contents


def write_off_chip_onnx_artifacts(model_dir, subgraphs_bcm, file_name_prefix, key_prefix, data_formats):
    """Write off-chip ONNX artifacts."""
    reference_dir = get_reference_dir(model_dir)
    ris_artifacts = {}
    for name in subgraphs_bcm:
        if 'off_chip' in name:
            filename = reference_dir / f"{file_name_prefix}{MUNC_ARTIFACT_DIRECTORY_NAME}_{name}.onnx"
            graph = create_offchip_graph(name, subgraphs_bcm, data_formats)
            graph.save(filename)
            ris_artifacts[f"{key_prefix}{name}"] = model_relative_path(filename, model_dir)
    return ris_artifacts


def create_tflite_offchip_graph(onnx_model, output_tflite_file):
    """Convert ONNX model to TensorFlow Lite model.

    This function converts @param onnx_model to a TensorFlow Lite model and writes it to @param output_tflite_file.
    """
    # Use MUNC to Fuse MUL nodes of the ONNX model.
    # Otherwise, onnx2tf will broadcast MUL nodes which is fused with FullyConnected layer to a two-dimension bias
    # tensor at the tflite graph but ArmNN only supports one-dimension bias tensor.
    sess = Session(onnx_model)
    sess.run_ops(ops.AbsorbOffchipMulNodes())

    # Check if the modified ONNX model is valid
    onnx.checker.check_model(onnx_model._model_proto)

    # Use a temporary directory
    with tempfile.TemporaryDirectory() as temp_directory:
        temp_directory = Path(temp_directory)

        # Modify those transpose nodes which are immediately after Softmax. The transpose was for
        # channel-first while onnx2tf change other nodes to channel-last which causes conflicts. Technically,
        # we should search transpose on the pattern of {softmax -> transpose -> conv} since onnx2tf only change Convs to
        # channel-last, so this is a quick fix for now.
        # Note: when Softmax's `axis` isn't the last dim, the v11->v13 up-converter wraps the
        # node as {... -> Softmax -> Reshape -> ...}. We walk through a single Reshape so the
        # {Softmax -> (Reshape ->)? Transpose} pattern is picked up in both cases.
        def _predecessor_through_reshape(n):
            pred = n.direct_predecessor(0)
            if pred is not None and pred.op_type == ONNXType.RESHAPE:
                pred = pred.direct_predecessor(0)
            return pred

        transposes = [node.name for node in onnx_model.get_nodes_with_op_type(ONNXType.TRANSPOSE)
                      if _predecessor_through_reshape(node) is not None
                      and _predecessor_through_reshape(node).op_type == ONNXType.SOFTMAX]
        # onnx2tf takes a replacement json file to modify nodes
        if len(transposes) > 0:
            # Change transpose permuation from [0, 3, 1, 2] to [0, 2, 1, 3] (swap channel first and
            # channel last.)
            transpose_params = {"param_target": "attributes", "param_name": "perm", "values": [0, 2, 1, 3]}
            replacement_json = {"format_version": 1, "operations": []}
            for transpose in transposes:
                replacement_json["operations"].append({**{"op_name": transpose}, **transpose_params})
            replacement_file = str(temp_directory / 'onnx2tf_replacement.json')
            with open(replacement_file, 'w') as f:
                json.dump(replacement_json, f)
        else:
            replacement_file = ''

        onnx_model_file = temp_directory / 'model.onnx'
        # Or save the modified ONNX model for debugging
        # onnx_model_file = output_tflite_file.with_suffix('.tf.onnx')
        onnx_model.save(onnx_model_file)
        onnx2tf_output_dir = str(temp_directory / 'saved_model')
        # onnx2tf makes global environment changes in load time, e.g. seeds random number generators. To avoid
        # its side effect it's better to run it as a separate process using onnx2tf CLI instead of using
        # `onnx2tf.convert`.
        subprocess.run(["onnx2tf",
                        # "--tflite_backend", "tf_converter",
                        "--output_keras_v3",
                        "--input_onnx_file_path", str(onnx_model_file),
                        "--output_folder_path", onnx2tf_output_dir,
                        "--param_replacement_file", replacement_file,
                        "--batch_size", "1",
                        "--not_use_onnxsim", "--not_use_opname_auto_generate", "--non_verbose"],
                       check=True)

        # Set memory growth to True for each GPU, otherwise TensorFlow will allocate all GPU memory and
        # never release it.
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        # Convert to tflite and write to output_tflite_file
        keras_model = tf_keras.models.load_model(onnx2tf_output_dir + '/model_float32_v3.keras')
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        with open(output_tflite_file, "wb") as f:
            f.write(tflite_model)


def write_off_chip_tflite_artifacts(model_dir, subgraphs_bcm, file_name_prefix, key_prefix, data_formats):
    """Write off-chip TFLite artifacts."""
    # TF may clobber CUDA_VISIBLE_DEVICES, so we save it and restore at the end.
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        reference_dir = get_reference_dir(model_dir)
        artifacts = {}
        for name in subgraphs_bcm:
            if 'off_chip' in name:
                filename = reference_dir / f"{file_name_prefix}{MUNC_ARTIFACT_DIRECTORY_NAME}_{name}.tflite"
                onnx_model = create_offchip_graph(name, subgraphs_bcm, data_formats)
                create_tflite_offchip_graph(onnx_model, filename)
                artifacts[f"{key_prefix}{name}"] = model_relative_path(filename, model_dir)
        return artifacts
    finally:
        # Restore CUDA_VISIBLE_DEVICES environment variable
        if cuda_visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']


def write_artifact_data(model_dir, artifact_data, input_names=()):
    """Write artifact data in h5 format to files in `model_dir`.

    Parameters
    ----------
    model_dir : str
        Path-like string pointing towards directory in which the artifact components will be saved.
    artifact_data : Dict[str, Any]
        Data to save as h5 files. See `_write_data_to_h5_file`.
    input_names : List[str]
        Model input names.

    Returns
    -------
    Dict
        Dictionary with the paths to the written files.
    """
    # Get the name of the input edge, i.e., input initializer for writing artifact data
    input_name = _get_model_input_name(input_names)
    data_files = {}
    # Write file for each output set
    for key, value in artifact_data.items():
        filename = model_dir / f'{MUNC_ARTIFACT_DIRECTORY_NAME}_{key}.h5'
        _write_data_to_h5_file(value, filename, input_name)
        data_files[f'{MUNC_ARTIFACT_DIRECTORY_NAME}_{key}'] = str(filename)
    return {'data_files': data_files}


def write_artifact_metadata(artifact_dir, metadata):
    """Write artifact content dictinary.

    Parameters
    ----------
    artifact_dir : Path
        Path to the artifact directory.
    metadata : Dict
        Artifact content dictionary.
    """
    with open(artifact_dir / 'contents.json', "w") as f:
        json.dump(metadata, f, indent=4)


def read_artifact_metadata(artifact_dir):
    """Read and return artifact content dictinary.

    Parameters
    ----------
    artifact_dir : Path
        Path to the artifact directory.

    Returns
    -------
    Dict
        Artifact content dictionary.
    """
    with open(artifact_dir / 'contents.json', "r") as f:
        return json.load(f)


class Artifact(AbstractContextManager):
    """A context manager that automates packing and unpacking Munc artifact files.

    It initializes an artifact directory, allows a caller to add files to it, and then packs the directory into
    an artifact tar.gz file on exit. In the scope of the context manager, the artifact directory is available as
    `artifact_dir` and the metadata dictionary is available as `metadata`.

    Example
    -------
    ```
    with artifact_writer.Artifact(out_file=artifact_tar_path):
        generate_standard_artifact_files(model,
                                         hw_config=hw_config,
                                         BGR=BGR,
                                         padding_value=padding_value,
                                         artifact_root_dir=artifact_root_dir,
                                         debug_dir=debug_dir)
    ```
    """

    def __init__(self, in_file=None, out_file=None, artifact_root_dir=None, keep_artifact_dir=True):
        """Initialize an instance.

        Parameters
        ----------
        in_file : Optional[PathLike]
            Path to an artifact tar.gz file to unpack into an artifact directory. If not specified, an empty artifact
            directory will be created.
        out_file : Optional[PathLike]
            Path to a target artifact tar.gz file. On exit, the artifact directory will be packed into this file.
            If not specified, the artifact directory will not be packed.
        artifact_root_dir : Optional[PathLike]
            Path to the root directory for the artifact. If not specified, a temporary directory will be used.
        keep_artifact_dir : Optional[bool]
            If True, the artifact directory will not be deleted on exit. By default True.
        """
        if in_file and not str(in_file).endswith(".tar.gz"):
            raise ValueError(f"Artifact tar path {in_file} does not include .tar.gz suffix.")
        if out_file and not str(out_file).endswith(".tar.gz"):
            raise ValueError(f"Artifact tar path {out_file} does not include .tar.gz suffix.")

        self.out_file = out_file
        self.keep_artifact_dir = keep_artifact_dir
        self.temp_artifact_root_dir = not artifact_root_dir
        if self.temp_artifact_root_dir:
            artifact_root_dir = tempfile.mkdtemp()

        try:
            self.artifact_root_dir = Path(artifact_root_dir)
            self.artifact_root_dir.mkdir(parents=True, exist_ok=True)
            self.artifact_dir = self.artifact_root_dir / MUNC_ARTIFACT_DIRECTORY_NAME

            if in_file is not None:
                logger.info(f'Unpacking a Munc artifact {in_file}')
                subprocess.run(["tar", "-C", str(artifact_root_dir), "-xf", str(in_file)], check=True)
                self.metadata = read_artifact_metadata(self.artifact_dir)
            else:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                self.metadata = {}
        except Exception as e:
            if self.temp_artifact_root_dir:
                shutil.rmtree(str(artifact_root_dir), ignore_errors=True)
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None and self.out_file:
                write_artifact_metadata(self.artifact_dir, self.metadata)
                logger.info(f'Packing a Munc artifact to {self.out_file}')
                subprocess.run(["tar", "-C", str(self.artifact_root_dir), "-cpzf", str(self.out_file),
                                MUNC_ARTIFACT_DIRECTORY_NAME],
                               check=True)
        finally:
            if self.temp_artifact_root_dir:
                shutil.rmtree(str(self.artifact_root_dir), ignore_errors=True)
            elif not self.keep_artifact_dir:
                shutil.rmtree(str(self.artifact_dir), ignore_errors=True)
        return False


OFF_CHIP_PROCESSOR_CLASSES = ['ONNXProcessor']  # 'ArmNNProcessor'


def write_pipeline(dest_file, artifact_dir, graphs_to_include, directory_contents, amp_address=0,
                   off_chip_processor_classes=OFF_CHIP_PROCESSOR_CLASSES,
                   absolute_paths=False):
    """Generate an inference engine pipeline configuration.

    Parameters
    ----------
    dest_file : PathLike
        A file to save the pipeline configuration to.
    artifact_dir : PathLike
        Path to a Munc artifact directory.
    graphs_to_include : List[str]
        List of graph names to include in the pipeline.
    directory_contents : Dict
        Munc artifact metadata.
    amp_address : int, optional
        An AMP address. Defaults to 0.
    off_chip_processor_classes : List[str], optional
        A lists of IE processors to use for off-chip graphs. Defaults to OFF_CHIP_PROCESSOR_CLASSES.
    absolute_paths : bool, optional
        If True, use absolute paths in the configuration. Defaults to False.
    """
    processor_config_constructors = {'ArmNNProcessor': _make_armnn_processor_config,
                                     'ONNXProcessor': _make_onnx_processor_config}

    def make_stage_config(graph_name):
        if "on_chip" in graph_name:
            classes = 'AMPProcessor'
            stage_config = _make_amp_processor_config(amp_address, artifact_dir, absolute_paths)
        else:
            classes = off_chip_processor_classes
            processor_configs = [processor_config_constructors[c](directory_contents, graph_name, artifact_dir,
                                                                  absolute_paths)
                                 for c in classes]
            stage_config = OmegaConf.merge(*processor_configs)
        stage_config['class'] = classes
        stage_config['name'] = graph_name
        return stage_config

    pipeline = {i + 1: make_stage_config(graph_name) for i, graph_name in enumerate(graphs_to_include)}
    contents_filtered = project(directory_contents, ['expected_input', 'padding_value', 'input_shapes'])
    cfg = OmegaConf.create()
    cfg['Pipeline'] = pipeline
    if len(contents_filtered) > 0:
        cfg['Contents'] = contents_filtered
    OmegaConf.save(cfg, dest_file)


def _make_amp_processor_config(address, artifact_dir, absolute_paths):
    runtime_yml = '../runtime.yml'
    if absolute_paths:
        runtime_yml = str((artifact_dir / runtime_yml).resolve())
    return OmegaConf.create({'class_cfg_data': {'address': address, 'runtime': runtime_yml}})


def _make_armnn_processor_config(directory_contents, graph_name, artifact_dir, absolute_paths):
    ie_tflite_artifacts = directory_contents["ie_tflite_artifacts"]
    tflite = ie_tflite_artifacts[graph_name]
    tflite_path = artifact_dir / tflite
    if absolute_paths:
        tflite = str(tflite_path.resolve())
    return OmegaConf.create({'class_cfg_data': {'filename': {'tflite': tflite}},
                             'io_name_map': _make_io_name_map(tflite_path)})


def _make_onnx_processor_config(directory_contents, graph_name, artifact_dir, absolute_paths):
    ie_artifacts = directory_contents["ie_artifacts"]
    onnx = ie_artifacts[graph_name]
    if absolute_paths:
        onnx = str((artifact_dir / onnx).resolve())
    return OmegaConf.create({'class_cfg_data': {'filename': {'onnx': onnx}}})


def _make_io_name_map(tflite_file):
    """Create input and output name mapping (io_name_map) since tflite graph doesn't preserve those names."""
    def io_name_map(all_details):
        return [{'tflite': details['name'], 'onnx': name} for name, details in all_details.items()]

    interpreter = Interpreter(model_path=str(tflite_file))
    signature_runner = interpreter.get_signature_runner()
    return io_name_map(signature_runner.get_input_details()) + io_name_map(signature_runner.get_output_details())
