import logging
import os
from pathlib import Path

from datasets import Dataset, load_dataset  # type: ignore

from vnnort import get_env_variable
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import VisualQuestionAnsweringInput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase
from vnnort.inference.evaluation.visual_question_answering import VisualQuestionAnsweringBenchmark

logger = logging.getLogger(__name__)
POPE_DATASET_PATH = get_env_variable("VNNORT_POPE_DATASET_PATH")
TEST_FILE_NAME = "pope-test-00000-of-00003.arrow"


def _find_test_file(dataset_path: str | Path) -> Path | None:
    root = Path(dataset_path)
    for path in root.rglob(TEST_FILE_NAME):
        return path  # returns the first match
    return None


class POPEDataset(DatasetBase):
    """Wrapper around the POPE visual question answering test dataset."""

    def __init__(self, dataset_path: str | Path = POPE_DATASET_PATH) -> None:
        """Initialize the dataset."""
        arrow_file_path = _find_test_file(dataset_path)
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
                self.dataset = load_dataset("lmms-lab/POPE", "default", cache_dir=dataset_path)["test"]

            finally:
                # Restore the original umask to avoid affecting other parts of the application
                os.umask(umask)

    def __getitem__(self, index: int) -> VisualQuestionAnsweringInput:
        """Return the 'index'th item of the dataset.

        Args:
            index (int): index of the item to return

        Returns:
            VisualQuestionAnsweringInput: Input to be returned.
        """
        data_dict = self.dataset[index]
        question = data_dict["question"] + " Answer with 'yes' or 'no'"
        result = VisualQuestionAnsweringInput(question=question, answer=data_dict["answer"], image=data_dict["image"])
        return result

    def __len__(self) -> int:
        """Return length of the dataset.

        Returns:
            int: length of the dataset
        """
        return len(self.dataset)

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return a subclass of BenchmarkBase which can be used to benchmark a model on this dataset.

        Returns:
            type[BenchmarkBase]: A subclass of BenchmarkBase
        """
        return VisualQuestionAnsweringBenchmark
