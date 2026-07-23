"""Artifact generation module."""
import logging

from munc._artifact import intermediate_outputs, graph_splitter, artifact_writer
from munc._artifact._prepare_off_on_chip_transitions import (collect_data_types_and_layouts,
                                                             make_get_onnx_layout_and_type)
from munc._artifact.write_ris_pipeline_artifacts import write_ris_pipeline_artifacts
from munc.graph_utils import get_op_type_count
from munc._constants import DEBUG_DIR, ONNXType
from munc._session import assert_model_is, MODELType


logger = logging.getLogger(__name__)

MUNC_ARTIFACT_VERSION = 2


def generate_standard_artifact_files(sess,
                                     artifact_dir,
                                     merge_subgraphs=True,
                                     padding_value=0,
                                     debug_dir=DEBUG_DIR,
                                     directory_contents=None,
                                     source_model=None):
    """Generate artifacts from a bcm-type model.

    Parameters
    ----------
    sess : munc.Session
        A session to use.
    artifact_dir : PathLike
        Artifact directory.
    merge_subgraphs : bool, optional
        Whether to merge topologically discontinuous offchip and onchip subgraphs
        (input and output portions separately), by default True.
    padding_value: int, optional
        A value to use for padding input images, by default 0.
    debug_dir : Optional[PathLike]
        A directory with debug information. If specified, it will be copied into the model directory.
    directory_contents : dict
        An artifact metadata (contents directory) to update.
    source_model : Optional[ONNXModel]
        An ONNX model that is the base model for this artifact. It will be included into the artifact.

    Raises
    ------
    ValueError
        If the provided model is not of type BCM."
    """
    model = sess.model
    assert_model_is(model, MODELType.COMPILER)

    logger.info('Writing artifacts...')
    expected_input = 'RGB' if model.get_meta_data('__BGR') is None else 'BGR'
    subgraphs_bcm = graph_splitter.split_onnx_graph(model, merge_subgroups=merge_subgraphs)

    metadata = artifact_writer.write_artifact(
        model_dir=artifact_dir,
        expected_input=expected_input,
        padding_value=padding_value,
        input_shapes={
            n: [d if isinstance(d, int) else -1 for d in s[1:]]
            for (n, s) in model.get_input_sizes().items()
        },
        source_model=source_model,
        op_type_count=get_op_type_count(model),
        subgraphs_bcm=subgraphs_bcm,
        debug_dir=debug_dir
    )
    directory_contents.update(metadata)

    generate_risv2_artifact_files(subgraphs_bcm, artifact_dir, directory_contents)
    generate_ie_artifact_files(subgraphs_bcm, artifact_dir, directory_contents)
    # generate_tflite_artifact_files(subgraphs_bcm, artifact_dir, directory_contents)
    generate_pipeline_artifact_files(subgraphs_bcm, artifact_dir, directory_contents)
    directory_contents['munc_artifact_version'] = MUNC_ARTIFACT_VERSION


def _get_pipeline_graphs(subgraphs_bcm):
    """Return a list of subgraph names that require a pipeline stage."""
    pipeline_graphs = [name for name, model in subgraphs_bcm.items()
                       if "on_chip" in name or _onnx_requires_pipeline_stage(model)]
    # Sort the graphs by the number at the end of the name, e.g. off_chip_0, on_chip_1, off_chip_2.
    return sorted(pipeline_graphs, key=lambda n: int(n[n.rindex('_') + 1:]))


def _onnx_requires_pipeline_stage(model):
    """Return true if the specified ONNX model needs to be handled by onnx-RT."""
    # Check for operations that need to be handled by onnx-RT
    # Standard CONV or GEMM layers
    if len(model.get_nodes_with_op_type([ONNXType.CONV, ONNXType.GEMM])) > 0:
        return True
    # Elementwise multiplication
    mul_nodes = model.get_nodes_with_op_type(ONNXType.MUL)
    initializers = set(model.get_initializer_names())
    return any([initializers.isdisjoint(set(mul_node.input)) for mul_node in mul_nodes])


def generate_risv2_artifact_files(subgraphs_bcm, artifact_dir, directory_contents):
    """Generate RISV2 off-chip ONNX files."""
    data_formats = collect_data_types_and_layouts(subgraphs_bcm)
    artifacts = artifact_writer.write_off_chip_onnx_artifacts(artifact_dir, subgraphs_bcm, "RISV2_", "RISV2_",
                                                              data_formats)
    reference_dir = artifact_writer.get_reference_dir(artifact_dir)

    pipeline_artifact_files = write_ris_pipeline_artifacts(reference_dir, directory_contents['onnx_graphs'],
                                                           artifacts)
    pipeline_artifacts = {fn.name: artifact_writer.model_relative_path(fn, artifact_dir)
                          for fn in pipeline_artifact_files}
    directory_contents['ris_artifacts'] = artifacts
    directory_contents['ris_pipeline_artifacts'] = pipeline_artifacts


def generate_ie_artifact_files(subgraphs_bcm, artifact_dir, directory_contents):
    """Generate Inference Engine off-chip ONNX files."""
    nhwc = make_get_onnx_layout_and_type("NHWC")
    data_formats = collect_data_types_and_layouts(subgraphs_bcm,
                                                  get_model_input_layout_and_type=nhwc,
                                                  get_model_output_layout_and_type=nhwc)
    artifacts = artifact_writer.write_off_chip_onnx_artifacts(artifact_dir, subgraphs_bcm, "ie_", "", data_formats)
    directory_contents['ie_artifacts'] = artifacts


def generate_tflite_artifact_files(subgraphs_bcm, artifact_dir, directory_contents):
    """Generate Inference Engine off-chip Tflite files."""
    nhwc = make_get_onnx_layout_and_type("NHWC")
    data_formats = collect_data_types_and_layouts(subgraphs_bcm,
                                                  get_edge_off_chip_layout_and_type=nhwc,
                                                  get_model_input_layout_and_type=nhwc,
                                                  get_model_output_layout_and_type=nhwc)
    artifacts = artifact_writer.write_off_chip_tflite_artifacts(artifact_dir, subgraphs_bcm, "ie_", "", data_formats)
    directory_contents['ie_tflite_artifacts'] = artifacts


def generate_pipeline_artifact_files(subgraphs_bcm, artifact_dir, directory_contents):
    """Generate Inference Engine pipeline definitions."""
    pipeline_graphs = _get_pipeline_graphs(subgraphs_bcm)
    pipeline_file = artifact_writer.get_reference_dir(artifact_dir) / "pipeline.yml"
    artifact_writer.write_pipeline(pipeline_file, artifact_dir, pipeline_graphs, directory_contents)


def generate_h5_artifact_files(sess,
                               artifact_dir=None,
                               image_numbers=(1, 5),
                               directory_contents=None):
    """Generate input/output data samples for each layer and write them to h5 files.

    Parameters
    ----------
    sess : munc.Session
        A session to use.
    artifact_dir : PathLike
        Artifact directory.
    image_numbers : tuple or None, optional
        Number of images used to generate intermediate outputs. If None, no intermediate outputs will be inferred and
        saved. By default (1, 5).
    directory_contents : dict
        An artifact metadata (contents directory) to update.
    """
    def get_image_sample_data(image_number):
        return intermediate_outputs.get_intermediate_outputs(sess, image_number)

    image_numbers = image_numbers if image_numbers is not None else []
    samples = {f'data_{image_number}': get_image_sample_data(image_number) for image_number in image_numbers}
    data_files_directory_contents = artifact_writer.write_artifact_data(artifact_dir, samples, input_names=())
    directory_contents.update(data_files_directory_contents)


def generate_artifact(sess,
                      artifact_dir,
                      image_numbers=(1, 5),
                      merge_subgraphs=True,
                      padding_value=0,
                      debug_dir=DEBUG_DIR,
                      directory_contents=None,
                      source_model=None
                      ):
    """Generate artifacts from a bcm-type model.

    Parameters
    ----------
    sess : munc.Session
        A session to use.
    artifact_dir : PathLike
        Artifact directory.
    image_numbers : tuple or None, optional
        Number of images used to generate intermediate outputs. If None, no intermediate outputs will be inferred and
        saved. By default (1, 5).
    merge_subgraphs : bool, optional
        Whether to merge topologically discontinuous offchip and onchip subgraphs
        (input and output portions separately), by default True.
    padding_value: int, optional
        A value to use for padding input images, by default 0.
    debug_dir : Optional[PathLike]
        A directory with debug information. If specified, it will be copied into the model directory.
    directory_contents : dict
        An artifact metadata (contents directory) to update.
    source_model : Optional[ONNXModel]
        A model this artifact is based on. If specified, it will be included into the artifact.

    Raises
    ------
    ValueError
        If the provided model is not of type BCM."
    """
    generate_standard_artifact_files(sess,
                                     merge_subgraphs=merge_subgraphs,
                                     padding_value=padding_value,
                                     artifact_dir=artifact_dir,
                                     debug_dir=debug_dir,
                                     directory_contents=directory_contents,
                                     source_model=source_model)

    generate_h5_artifact_files(sess,
                               image_numbers=image_numbers,
                               artifact_dir=artifact_dir,
                               directory_contents=directory_contents)

    artifact_writer.write_artifact_metadata(artifact_dir, directory_contents)
