import os
from pathlib import Path

from datasets import Dataset, load_dataset  # type: ignore

from vnnort import get_env_variable
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import QuestionAnsweringInput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase

SQUAD_DATASET_PATH = get_env_variable("VNNORT_SQUAD_DATASET_PATH")
VALIDATION_FILE_NAME = "squad-validation.arrow"


def _find_validation_file(dataset_path: str | Path) -> Path | None:
    root = Path(dataset_path)
    for path in root.rglob(VALIDATION_FILE_NAME):
        return path  # returns the first match
    return None


class SquadDataset(DatasetBase):
    """Wrapper around the squad dataset."""

    def __init__(self, dataset_path: str = SQUAD_DATASET_PATH) -> None:
        """Initialize the squad dataset.

        We utilize the huggingface datasets library. This automatically downloads the datasets to the
        dataset path.
        """
        arrow_file_path = _find_validation_file(dataset_path)
        if arrow_file_path is not None:
            # In case the file already exists, we load it locally.
            # This avoids any API calls made to huggingface
            self.dataset = Dataset.from_file(str(arrow_file_path))

        else:
            try:
                # Load the dataset, which internally creates lock files.
                # With umask set to 0o000, these files will be created with full read and write permissions for all users.
                # Temporarily set umask to 0o000, ensuring newly created files have full rw-rw-rw- (666) permissions
                umask = os.umask(0o000)
                self.dataset = load_dataset("rajpurkar/squad", split="validation", cache_dir=dataset_path)

            finally:
                # Restore the original umask to avoid affecting other parts of the application
                os.umask(umask)

    def __getitem__(self, index: int) -> QuestionAnsweringInput:
        """Return the sample at index.

        Args:
            index (int): index to access. Needs to be 0 < index < len(self)

        Returns:
            QuestionAnsweringInput: the sample at index.
        """
        data_dict = self.dataset[index]
        data = QuestionAnsweringInput(
            question=data_dict["question"],
            context=data_dict["context"],
            answers=data_dict["answers"],
        )
        return data

    def __len__(self) -> int:
        """Return the number of samples of the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return a function which can be used to benchmark a model on this dataset."""
        from vnnort.inference.evaluation.question_answering import QuestionAnsweringBenchmark

        return QuestionAnsweringBenchmark
