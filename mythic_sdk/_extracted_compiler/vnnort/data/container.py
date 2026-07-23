"""This module defines several data container formats, which are used in the inference data flow.

All datasets need to return an instance of InputData. Image classification datasets will return an instance of
ImageClassificationData. Image detection datasets will return an instance of ImageDetectionData.

All models preprocessing functions take these as input and the output of the postprocess function will be an
instance of OutputData. Image detection models will return an instance of ImageDetectionOutput and so on.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional

import PIL.Image
import numpy as np
from numpy.typing import NDArray


@dataclass
class InputData(ABC):
    """Common base class for all input data container formats."""

    pass


@dataclass
class OutputData(ABC):
    """Common base class for all output data container formats."""

    pass


# Image Classification ################
@dataclass
class ImageClassificationInput(InputData):  # noqa: DOC601,DOC603
    """Data class for image classification.

    Parameters:
        image (PIL.Image.Image): The image to classify.
        label (int): The label to classify the image with.
    """

    image: PIL.Image.Image
    label: int

    def __post_init__(self) -> None:
        """Check if the image and label are of the correct type."""
        if not isinstance(self.image, PIL.Image.Image):
            raise ValueError("image must be of type PIL.Image.Image")
        if not isinstance(self.label, int):
            raise ValueError("label must be of type int")


@dataclass
class ImageClassificationOutput(OutputData):  # noqa: DOC601,DOC603
    """Output data class for image classification.

    Parameters:
        logits (NDArray[np.float32]): N dimensional vector.
        probabilities (NDArray[np.float32]): N dimensional vector.
        label (int): The classification label.
    """

    logits: NDArray[np.float32]
    probabilities: NDArray[np.float32]
    label: int

    def __post_init__(self) -> None:
        """Check if the logits, probabilities and label are of the correct type and shape."""
        if not isinstance(self.logits, np.ndarray):
            raise ValueError("logits must be of type np.ndarray")
        if not isinstance(self.probabilities, np.ndarray):
            raise ValueError("probabilities must be of type np.ndarray")
        if not isinstance(self.label, int):
            raise ValueError("label must be of type int")

        if not 0 <= self.label < self.logits.shape[0]:
            raise ValueError("label must be in range [0, logits.shape[0])")

        if not len(self.logits.shape) == 1:
            raise ValueError("logits must be of shape (N,)")

        if not len(self.probabilities.shape) == 1:
            raise ValueError("probabilities must be of shape (N,)")

        if not self.probabilities.shape[0] == self.logits.shape[0]:
            raise ValueError("probabilities and logits must have the same length")


# Image Detection ################
@dataclass
class ImageDetectionInput(InputData):  # noqa: DOC601,DOC603
    """Data class for image detection.

    Parameters:
        image (PIL.Image.Image): The image to detect.
        boxes (Optional[NDArray[np.float32]]): N x 4 array in format xyxy.
        labels (Optional[NDArray[np.int32]]): N dimensional array of labels.
    """

    image: PIL.Image.Image
    boxes: Optional[NDArray[np.float32]]
    labels: Optional[NDArray[np.int32]]

    def __post_init__(self) -> None:
        """Check if the image, boxes and labels are of the correct type and shape."""
        if not isinstance(self.image, PIL.Image.Image):
            raise ValueError("image must be of type PIL.Image.Image")
        if self.boxes is not None and self.labels is not None:
            if not isinstance(self.boxes, np.ndarray):
                raise ValueError("boxes must be of type np.ndarray")
            if not isinstance(self.labels, np.ndarray):
                raise ValueError("labels must be of type np.ndarray")

            if not len(self.boxes.shape) == 2:
                raise ValueError("boxes must be of shape (N, 4)")

            if not len(self.labels.shape) == 1:
                raise ValueError("labels must be of shape (N,)")

            if not self.boxes.shape[1] == 4:
                raise ValueError("boxes must be of shape (N, 4)")

            if not self.boxes.shape[0] == self.labels.shape[0]:
                raise ValueError("boxes and labels must have the same length")


@dataclass
class ImageDetectionOutput(OutputData):  # noqa: DOC601,DOC603
    """Output data class for image detection.

    Parameters:
        boxes (NDArray[np.float32]): N x 4 array in format xyxy.
        scores (NDArray[np.float32]): N dimensional array of confidence scores.
        labels (NDArray[np.int32]): N dimensional array of labels.
    """

    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    labels: NDArray[np.int32]

    def __post_init__(self) -> None:
        """Check if the boxes, scores and labels are of the correct type and shape."""
        if not isinstance(self.boxes, np.ndarray):
            raise ValueError("boxes must be of type np.ndarray")
        if not isinstance(self.scores, np.ndarray):
            raise ValueError("scores must be of type np.ndarray")
        if not isinstance(self.labels, np.ndarray):
            raise ValueError("labels must be of type np.ndarray")

        if not len(self.boxes.shape) == 2:
            raise ValueError("boxes must be of shape (N, 4)")

        if not len(self.scores.shape) == 1:
            raise ValueError("scores must be of shape (N,)")

        if not len(self.labels.shape) == 1:
            raise ValueError("labels must be of shape (N,)")

        if not self.boxes.shape[0] == self.scores.shape[0] == self.labels.shape[0]:
            raise ValueError("boxes, scores and labels must have the same length")

        if not self.boxes.shape[1] == 4:
            raise ValueError("boxes must be of shape (N, 4)")


@dataclass
class ImageSegmentationInput(InputData):  # noqa: DOC601,DOC603
    """Data class for image segmentation input.

    Parameters:
        image (PIL.Image.Image): The image to segment.
        segmentation_map (Optional[NDArray[np.int32]]): The segmentation map, if available.
    """

    image: PIL.Image.Image
    segmentation_map: Optional[NDArray[np.int32]]

    def __post_init__(self) -> None:
        """Validate attributes of the input data class."""
        if not isinstance(self.image, PIL.Image.Image):
            raise ValueError("image must be of type PIL.Image.Image")

        if self.segmentation_map is not None:
            if not isinstance(self.segmentation_map, np.ndarray):
                raise ValueError("segmentation_map must be of type np.ndarray")
            if self.segmentation_map.ndim != 2:
                raise ValueError("segmentation_map must be a 2D array")
            if self.image.size[::-1] != self.segmentation_map.shape:
                raise ValueError("image size and segmentation_map need to have same shape")


@dataclass
class ImageSegmentationOutput:  # noqa: DOC601,DOC603
    """Output data class for image segmentation.

    Parameters:
        segmentation_map (NDArray[np.float32]): The resulting segmentation map.
    """

    segmentation_map: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate attributes of the output data class."""
        if not isinstance(self.segmentation_map, np.ndarray):
            raise ValueError("segmentation_map must be of type np.ndarray")
        if self.segmentation_map.ndim != 2:
            raise ValueError("segmentation_map must be a 2D array")


# Multi-View 3D Detection ################
@dataclass
class MultiViewDetection3DInput(InputData):  # noqa: DOC601,DOC603
    """Data class for surround-view 3D detection (e.g. BEVFormer on nuScenes).

    The container holds everything that is common to BEVFormer-tiny / -small / -base; the
    per-variant preprocess class is responsible for image resizing, normalization, padding,
    and the matching `lidar2img` row scaling.

    Parameters:
        images (list[PIL.Image.Image]): The N camera images for a single timestep, in a
            stable order chosen by the dataset (typically 6 for nuScenes).
        lidar2img (NDArray[np.float32]): (N, 4, 4) projection matrices built from
            intrinsics/extrinsics at native image resolution.
        can_bus (NDArray[np.float32]): (18,) ego-motion vector. Positions [:3] and [-1]
            already hold the delta to the previous frame in the same scene (zeros when
            `is_first_in_scene`); the remaining entries are absolute (quaternion, yaw
            radians, etc.) — mirrors BEVFormer's `TemporalState.update_can_bus_delta`.
        is_first_in_scene (bool): True when this sample starts a new scene; consumers
            use it to reset prev_bev / use_prev_bev model state.
        sample_token (str): nuScenes sample (keyframe) token.
        scene_token (str): nuScenes scene token.
        gt_boxes (Optional[NDArray[np.float32]]): (M, 9) ground-truth 3D boxes in LiDAR
            frame, when annotations are available. None for splits without GT.
        gt_labels (Optional[NDArray[np.int32]]): (M,) class indices into the nuScenes
            detection class list, aligned with `gt_boxes`.
        gt_names (Optional[list[str]]): (M,) class names, aligned with `gt_boxes`.
    """

    images: list[PIL.Image.Image]
    lidar2img: NDArray[np.float32]
    can_bus: NDArray[np.float32]
    is_first_in_scene: bool
    sample_token: str
    scene_token: str
    gt_boxes: Optional[NDArray[np.float32]] = None
    gt_labels: Optional[NDArray[np.int32]] = None
    gt_names: Optional[list[str]] = None

    def __post_init__(self) -> None:
        """Validate shapes/types and the GT length invariant."""
        if not isinstance(self.images, list) or not all(isinstance(im, PIL.Image.Image) for im in self.images):
            raise ValueError("images must be a list of PIL.Image.Image")
        n_cams = len(self.images)
        if not isinstance(self.lidar2img, np.ndarray) or self.lidar2img.shape != (n_cams, 4, 4):
            raise ValueError(f"lidar2img must be an ndarray of shape ({n_cams}, 4, 4)")
        if not isinstance(self.can_bus, np.ndarray) or self.can_bus.shape != (18,):
            raise ValueError("can_bus must be an ndarray of shape (18,)")
        if not isinstance(self.is_first_in_scene, bool):
            raise ValueError("is_first_in_scene must be a bool")
        gt_fields = (self.gt_boxes, self.gt_labels, self.gt_names)
        if any(f is None for f in gt_fields) and any(f is not None for f in gt_fields):
            raise ValueError("gt_boxes / gt_labels / gt_names must either all be None or all be set")
        if self.gt_boxes is not None:
            assert self.gt_labels is not None and self.gt_names is not None  # for type checkers
            if self.gt_boxes.ndim != 2 or self.gt_boxes.shape[1] != 9:
                raise ValueError("gt_boxes must be of shape (M, 9)")
            if self.gt_labels.shape != (self.gt_boxes.shape[0],):
                raise ValueError("gt_labels must be of shape (M,) matching gt_boxes")
            if len(self.gt_names) != self.gt_boxes.shape[0]:
                raise ValueError("gt_names must have length M matching gt_boxes")


@dataclass
class MultiViewDetection3DOutput(OutputData):  # noqa: DOC601,DOC603
    """Output for multi-view 3D detection (BEVFormer-style).

    Parameters:
        boxes (NDArray[np.float32]): (N, 9) per-box state — (x, y, z, w, l, h, yaw, vx, vy)
            in the same frame as the matching `MultiViewDetection3DInput.gt_boxes`
            (LiDAR frame for the BEVFormer pipeline).
        scores (NDArray[np.float32]): (N,) confidence scores in [0, 1].
        labels (NDArray[np.int32]): (N,) class indices into the dataset's class list.
    """

    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    labels: NDArray[np.int32]

    def __post_init__(self) -> None:
        """Validate shapes / dtypes."""
        for name, val in (("boxes", self.boxes), ("scores", self.scores), ("labels", self.labels)):
            if not isinstance(val, np.ndarray):
                raise ValueError(f"{name} must be an ndarray")
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 9:
            raise ValueError("boxes must be of shape (N, 9)")
        n = self.boxes.shape[0]
        if self.scores.shape != (n,):
            raise ValueError("scores must be of shape (N,) matching boxes")
        if self.labels.shape != (n,):
            raise ValueError("labels must be of shape (N,) matching boxes")


# Question Answering ################
@dataclass
class QuestionAnsweringInput(InputData):  # noqa: DOC601,DOC603
    """Input dataclass for question answering.

    Parameters:
        question (str): Question string.
        context (str): Context string.
        answers (Optional[list[str]]): List of possible answers.
    """

    question: str
    context: str
    answers: Optional[list[str]] = None


@dataclass
class QuestionAnsweringOutput(OutputData):  # noqa: DOC601,DOC603
    """Output dataclass for question answering.

    Parameters:
        answer (str): The generated answer text.
        score (float): Confidence score for the answer.
        start (int): Start position of answer in context.
        end (int): End position of answer in context.
    """

    answer: str
    score: float
    start: int
    end: int


# Text Generation #################
@dataclass
class TextGenerationInput(InputData):  # noqa: DOC601,DOC603
    """Input dataclass for text generation.

    Parameters:
        input_text (str): Input string for text generation.
        expected_text (str): The groundtruth expected text provided by a dataset.
        generated_answer (str): The answer generated so far by the model.
    """

    input_text: str
    expected_text: str

    generated_answer: str = ""

    def __post_init__(self) -> None:
        """Validate attributes of the text generation input data class."""
        if not isinstance(self.input_text, str):
            raise ValueError("input_text needs to be of type str")
        if not isinstance(self.generated_answer, str):
            raise ValueError("generated_answer needs to be of type str")
        if not isinstance(self.expected_text, str):
            raise ValueError("generated_answer needs to be of type str")


@dataclass
class TextGenerationOutput(OutputData):  # noqa: DOC601,DOC603
    """Output dataclass for text generation.

    Parameters:
        output_text (str): Output string from text generation.
    """

    output_text: str

    def __post_init__(self) -> None:
        """Validate attributes of the text generation output data class."""
        if not isinstance(self.output_text, str):
            raise ValueError("output_text needs to be of type str")


@dataclass
class VisualQuestionAnsweringInput(InputData):
    """Input dataclass for visual question answering."""

    question: str
    image: PIL.Image.Image
    answer: str
    generated_answer_str: str = ""

    def __post_init__(self) -> None:
        """Validate attributes of the text generation input data class."""
        if not isinstance(self.question, str):
            raise ValueError("input_text needs to be of type str")
        if not isinstance(self.image, PIL.Image.Image):
            raise ValueError("input_text needs to be of type str")
        if not isinstance(self.answer, str):
            raise ValueError("input_text needs to be of type str")
        if not isinstance(self.generated_answer_str, str):
            raise ValueError("input_text needs to be of type str")


@dataclass
class VisualQuestionAnsweringOutput(OutputData):  # noqa: DOC601,DOC603
    """Output dataclass for text generation.

    Parameters:
        answer (str): Answer of the model
    """

    answer: str

    def __post_init__(self) -> None:
        """Validate attributes of the visual question answering output data class."""
        if not isinstance(self.answer, str):
            raise ValueError("input_text needs to be of type str")
