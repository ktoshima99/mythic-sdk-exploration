"""Helper functions for adapting the run_rm*r.py command line interface to Hydra."""

from pathlib import Path
from contextlib import suppress


def rewrite_argv(argv, config_file_arg="--inference-config", add_wandb_entity=True):
    """Rewrite the standard run_rmir.py command line parameters as Hydra key overrides.

    This is currently needed for compatibility with rmclient.
    """
    argv = list(argv)

    config_path = None
    if config_file_arg:
        with suppress(ValueError):
            pos = argv.index(config_file_arg)
            argv.pop(pos)
            config_path = Path(argv.pop(pos)).absolute()

    extra_keys = []
    try:
        pos = argv.index("--test")
        argv.pop(pos)
    except ValueError:
        if add_wandb_entity:
            extra_keys.append("++wandb.entity=${entity_prod}")

    to_conf_keys = {
        "--run-date": "run_date",
        "--build-id": "build_id",
        "--build-user": "build_user",
        "--input-artifact": "src"
    }

    for arg, key in to_conf_keys.items():
        try:
            pos = argv.index(arg)
            argv.pop(pos)
            val = argv.pop(pos)
            extra_keys.append(f'++{key}="{val}"')
        except ValueError:
            pass

    config_options = ["-cn", config_path.name, "-cp", str(config_path.parent)] if config_path else []
    return [argv[0]] + config_options + argv[1:] + extra_keys
