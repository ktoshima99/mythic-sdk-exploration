import sys
from pathlib import Path

from vnnort import configure_logging

if __name__ == "__main__":
    # Python sets __package__=None when run as __main__, making relative imports fail.
    # Insert this script's directory so mythic_utils and the sibling bevformer package
    # can be imported by name both from this in-repo location and from the released
    # `scripts/` directory. Done at runtime (inside __main__) rather than at import time
    # so that the recursive `vnnort.models` package walk does not re-enter
    # `bevformer.bevformer_tiny` mid-initialization.
    sys.path.insert(0, str(Path(__file__).parent))

    from bevformer.bevformer_tiny import BevformerTiny
    from mythic_utils import parse_arguments, run_vnn_flow

    args = parse_arguments()

    # Prepare/Check input/output paths
    result_directory: Path = args.result_directory
    result_directory.mkdir(parents=True, exist_ok=True)

    # --source_onnx is accepted for CLI parity with the other Mythic scripts but is
    # unused: BevformerTiny builds its ONNX from onnxscript and does not load a file.

    configure_logging()

    # Run v-NN ORT pipeline with this model
    model = BevformerTiny(result_directory)
    run_vnn_flow(
        model,
        result_directory,
        system_config=Path(__file__).parent / "system_configs" / "bevformer.cfg",
        skip_validation=True,
        advanced=True,
    )
