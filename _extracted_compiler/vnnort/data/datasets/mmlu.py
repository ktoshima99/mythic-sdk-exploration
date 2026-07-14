import logging
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset  # type: ignore

from vnnort import configure_logging, get_env_variable
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import TextGenerationInput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase
from vnnort.inference.evaluation.text_generation import TextGenerationBenchmark

logger = logging.getLogger(__name__)
MMLU_DATASET_PATH = get_env_variable("VNNORT_MMLU_DATASET_PATH")
TEST_FILE_NAME = "mmlu-test.arrow"
N_EXAMPLE_PROMPTS = 5  # Each input is preprended with 5 example question and answer pairs.

# Some of this code is based on https://github.com/hendrycks/test
subjects = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
choice_letters = ["A", "B", "C", "D"]


def _find_test_file(dataset_path: str) -> Path | None:
    root = Path(dataset_path)
    for path in root.rglob(TEST_FILE_NAME):
        return path  # returns the first match
    return None


def _format_subject(subject: str) -> str:
    entries = subject.split("_")
    s = ""
    for entry in entries:
        s += " " + entry
    return s


def _format_example(data: dict[str, Any], include_answer: bool) -> str:
    prompt = str(data["question"])
    possible_answers = data["choices"]
    assert len(possible_answers) == 4

    for letter, choice in zip(choice_letters, possible_answers):
        prompt += f"\n{letter}. {choice}"
    prompt += "\nAnswer:"
    if include_answer:
        answer_index = int(data["answer"])
        answer = choice_letters[answer_index]
        prompt += f" {answer}\n\n"
    return prompt


def _generate_prompt(example_data: list[dict[str, Any]], data: dict[str, Any]) -> str:
    """Generate input prompt for model."""
    subject = data["subject"]
    subject = _format_subject(subject)
    prompt = f"The following are multiple choice questions (with answers) about {subject}. Answer only with a single letter!\n\n"

    for entry in example_data:
        prompt += _format_example(entry, include_answer=True)

    prompt += _format_example(data, include_answer=False)
    return prompt


class MMLUDataset(DatasetBase):
    """Wrapper around the MMLU test dataset."""

    def __init__(self, dataset_path: str = MMLU_DATASET_PATH) -> None:
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
                self.dataset = load_dataset("cais/mmlu", "all")["test"]

            finally:
                # Restore the original umask to avoid affecting other parts of the application
                os.umask(umask)

        # Prepare data sorted by subject
        data_dict: dict[str, list[dict[str, Any]]] = {}  # subject: list[data-dict]
        for entry in self.dataset:
            subject = entry["subject"]
            if subject not in data_dict:
                data_dict[subject] = []
            data_dict[subject].append(entry)

        # For all subjects, reserve the 5 first entries as example preprended to each prompt
        prompts = []
        answers = []
        for entries in data_dict.values():
            example_data = entries[:N_EXAMPLE_PROMPTS]
            for entry in entries[N_EXAMPLE_PROMPTS:]:
                prompt = _generate_prompt(example_data, entry)
                answer = choice_letters[entry["answer"]]
                prompts.append(prompt)
                answers.append(answer)
        self.prompts = prompts
        self.answers = answers

    def __getitem__(self, index: int) -> TextGenerationInput:
        """Return the 'index'th item of the dataset.

        Args:
            index (int): index of the item to return

        Returns:
            TextGenerationInput: Input to be returned.
        """
        prompt = self.prompts[index]
        answer = self.answers[index]

        return TextGenerationInput(input_text=prompt, expected_text=answer)

    def __len__(self) -> int:
        """Return length of the dataset.

        Returns:
            int: length of the dataset
        """
        return len(self.prompts)

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return a subclass of BenchmarkBase which can be used to benchmark a model on this dataset.

        Returns:
            type[BenchmarkBase]: A subclass of BenchmarkBase
        """
        return TextGenerationBenchmark


if __name__ == "__main__":
    configure_logging("DEBUG")
    ds = MMLUDataset()
    entry = ds[0]
    ...
