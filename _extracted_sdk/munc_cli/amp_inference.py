"""Helper functions for benchmarking inference on silicon."""

from tempfile import TemporaryDirectory
from datetime import datetime

from omegaconf import OmegaConf

from munc.cli.helpers import resolve_function
from mythic.model_deployment.rmcr.firmware_artifact import load_artifact
from mythic.model_deployment.rmir.infer import RISPipelineContextManager
from mythic.model_deployment.rmir.common import save_run_json
from mythic.model_deployment.rmir.LxL_gather import lxl_setup_and_gather


def make_eval_config(cfg, include_sections=('inference',)):
    """Copy the specified config sections into a new config object and make configuration keys uppercase.

    This is for compatibility with retrain-tools.
    """
    def copy_section(s):
        return {k.upper(): v for k, v in s.items()}

    return OmegaConf.create({s.upper(): copy_section(cfg[s]) for s in include_sections})


def benchmark_amp_inference(cfg, input_artifact=None):
    """Evaluate a firmware artifact on silicon, log computed metrics."""
    eval_cfg = make_eval_config(cfg)
    input_artifact = input_artifact or cfg.src
    validator_factory = cfg.factory
    make_validator = resolve_function(validator_factory, 'validator factory')
    validator = make_validator(cfg)

    with TemporaryDirectory() as artifact_dir:
        load_artifact(input_artifact, artifact_dir)
        with RISPipelineContextManager(eval_cfg.INFERENCE, artifact_dir) as ris_inference:
            metrics = validator(ris_inference, eval_cfg.INFERENCE.MAX_SAMPLES)

        if cfg.validate:
            log_data = {**metrics, **ris_inference.runtime_data}

            save_run_json(json_file=cfg.log_file, metrics_dict=log_data, args=OmegaConf.to_container(cfg),
                          cfg=OmegaConf.to_container(eval_cfg))
        else:
            log_data = ris_inference.runtime_data

        return metrics, log_data


def collect_data_for_layer_analysis(inference_cfg, wandb_cfg):
    """Evaluate a firmware artifact on silicon, save layer input and output samples for Layer Analysis."""
    eval_cfg = make_eval_config(inference_cfg, ('inference', 'lxl'))
    input_artifact = inference_cfg.src
    validator_factory = inference_cfg.factory
    make_validator = resolve_function(validator_factory, 'validator factory')
    validator = make_validator(inference_cfg)

    run_date = datetime.strptime(inference_cfg.run_date, "%Y%m%d_%H%M%S") if inference_cfg.run_date else datetime.now()
    with TemporaryDirectory() as artifact_dir:
        lxl_setup_and_gather(input_artifact=input_artifact, artifact_dir=artifact_dir, cfg=eval_cfg,
                             json_metadata=dict(args=OmegaConf.to_container(inference_cfg),
                                                cfg=OmegaConf.to_container(eval_cfg)),
                             build_id=inference_cfg.build_id, run_date=run_date, do_infer_func=validator,
                             wandb_init_keys=wandb_cfg, model=inference_cfg.model)
