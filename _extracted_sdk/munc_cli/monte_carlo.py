"""Helpers for running Monte Carlo evaluation sweeps from the CLI."""

import json
import logging
from collections import defaultdict
import os

from cloudpathlib import AnyPath
from hydra.utils import instantiate
from funcy import omit
from omegaconf import OmegaConf
import numpy as np

from munc._monte_carlo.chip_instance_generator import random_model_instances, get_schedule_num_samples
from munc._monte_carlo.tolerance import compute_lower_tolerance
from munc.cli.helpers import record_model_metrics


logger = logging.getLogger(__name__)


def get_num_samples(config, first_schedule_step=0):
    """Return the number of samples to run based on the config."""
    schedule = _parse_schedule(config.get('schedule'))
    default = get_schedule_num_samples(schedule[first_schedule_step:])
    num_samples = config.get('num_samples', default)
    return num_samples if first_schedule_step == 0 else min(num_samples, default)


def _parse_schedule(schedule, priority_key='priority'):
    """Convert `schedule` to a format expected by `random_model_instances`."""
    # Create a sorted list of steps (lower values = higher priority).
    schedule = list(sorted(schedule.values(), key=lambda config: config[priority_key]))
    # Instantiate weight_randomizer (convert from a function name to a function).
    schedule = instantiate(schedule)
    # Drop OmegaConf wrappers.
    schedule = OmegaConf.to_container(schedule, resolve=True)
    # Drop the priority_key, because random_model_instances does not expect it.
    schedule = [omit(step, priority_key) for step in schedule]
    return schedule


def collect_accuracy_data(config, output_dir, sess, run_evaluator, start_index=0):
    """Run evaluator on randomized session instances and store metrics to disk.

    Parameters
    ----------
    config : dict
        Configuration with Monte Carlo parameters including `num_samples`, `schedule`, and `model_type`.
    `schedule` is a dict of randomization steps. Each step is a dict.
    output_dir : str or os.PathLike
        Directory where metrics JSON files are written.
    sess : Session
        Base session used to generate randomized model instances.
    run_evaluator : Callable[[dict, Session], dict]
        Function that executes evaluation for a single sample and returns metric values.
    start_index : Optional[int]
        The first index to start numbering output files from.
    """
    output_dir = AnyPath(output_dir)
    num_samples = get_num_samples(config)
    randomization_schedule = _parse_schedule(config.get('schedule'))
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_str = f" on GPU {gpu}" if gpu and "," not in gpu else ""
    for i, s in enumerate(random_model_instances(sess, randomization_schedule, num_tests=num_samples)):
        output_file = output_dir / f'metrics_{i + start_index:04d}.json'
        cfg = dict(config)
        cfg['metrics_file'] = str(output_file)
        logger.info(f'Running evaluator for sample {i + 1}/{num_samples}{gpu_str}, writing metrics to {output_file}')
        metrics = run_evaluator(cfg, s)
        record_model_metrics(cfg, config.get("model_type"), metrics)


def load_accuracy_data(metrics_dirs, model_key):
    """Load metrics files produced by `collect_accuracy_data`.

    Parameters
    ----------
    metrics_dirs : List[str|os.PathLike]|str|os.PathLike
        Directory containing metrics JSON files named `metrics_*.json`.
    model_key : str
        Key under which metrics are stored in each JSON record.

    Returns
    -------
    dict
        Mapping of metric name to a list of observed values across samples.
    """
    results = defaultdict(list)
    for metrics_dir in [metrics_dirs] if isinstance(metrics_dirs, (str, os.PathLike)) else metrics_dirs:
        metrics_dir = AnyPath(metrics_dir)
        for metrics_file in metrics_dir.glob('metrics_[0-9]*.json'):
            with metrics_file.open('r') as f:
                metrics = json.load(f)[model_key]
                for key, value in metrics.items():
                    results[key].append(value)
    return results


def process_accuracy_data(accuracy_data, prop, confidence, metric_keys=None, include_mean_std=False):
    """Compute lower confidence bounds for accuracy metrics.

    Parameters
    ----------
    accuracy_data : dict
        Mapping of metric name to a list of observed values.
    prop : float
        Desired proportion of the distribution to cover.
    confidence : float
        Statistical confidence level for the lower tolerance bound.
    metric_keys : iterable of str, optional
        Specific metrics to process. Defaults to all keys in `accuracy_data`.

    Returns
    -------
    dict
        Mapping of metric name to lower tolerance bounds.
    """
    if metric_keys is None:
        metric_keys = accuracy_data.keys()
    accuracies = {key: compute_lower_tolerance(accuracy_data[key], prop, confidence) for key in metric_keys}
    if include_mean_std:
        accuracies = (accuracies
                      | {key + " mean": np.mean(accuracy_data[key]) for key in metric_keys}
                      | {key + " std": np.std(accuracy_data[key]) for key in metric_keys})
    return accuracies
