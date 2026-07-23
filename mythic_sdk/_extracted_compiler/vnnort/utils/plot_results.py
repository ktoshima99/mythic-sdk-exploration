import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vnnort.data.container import ImageDetectionOutput
from vnnort.data.datasets.coco import CocoClassRemapper

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def plot_detection_results_COCO(
    pil_img: Image.Image, detections: ImageDetectionOutput, result_path: str | None = None
) -> None:
    """
    Plot detection results on a image using COCO classes.

    Args:
        pil_img (Image.Image): The input image to display detections on.
        detections (ImageDetectionOutput): Detection results containing scores, bounding boxes, and labels.
        result_path (str | None): Path to save the resulting image. If None, the plot is displayed instead.

    Returns:
        None: Nothing is returned.
    """
    prob = detections.scores
    boxes = detections.boxes
    labels = detections.labels

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    class_remapper = CocoClassRemapper()
    for p, (xmin, ymin, xmax, ymax), c, label in zip(prob, boxes.tolist(), colors, labels):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))  # type: ignore
        # cl = p.argmax()
        # print(cl,CLASSES[cl])
        int_label = int(label)
        class_name = class_remapper.clean_label_to_name(int_label)
        text = f"{class_name}: {p:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")

    if result_path is None:
        plt.show()
    else:
        plt.savefig(result_path)


def overlay_boxes_and_labels(pil_img: Image.Image, detections: ImageDetectionOutput, threshold: float = 0.2) -> None:
    """
    Overlay bounding boxes and labels on an image.

    Args:
        pil_img (Image.Image): The input image to overlay detections on.
        detections (ImageDetectionOutput): Detection results containing bounding boxes, labels, and scores.
        threshold (float): Minimum confidence score for a detection to be displayed. Defaults to 0.2.

    Returns:
        None: Nothing is returned.
    """
    # Load and prepare the image
    img_array = np.array(pil_img)
    labels = detections.labels
    boxes = detections.boxes
    scores = detections.scores
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 7))
    ax.imshow(img_array)

    # Define a list of colors for different labels (you can customize this list)
    colors = ["r", "g", "b", "c", "m", "y", "k"]

    # Overlay the detection boxes and labels
    for i in range(len(labels)):
        label = labels[i]
        det = boxes[i]
        score = scores[i]
        if score < threshold:  # Skip if the score is zero
            continue

        x1, y1, x2, y2 = det
        width, height = x2 - x1, y2 - y1

        # Create a rectangle patch
        rect = matplotlib.patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor=colors[label % len(colors)], facecolor="none"
        )

        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add label text
        plt.text(
            x1,
            y1 - 10,
            f"Label: {label} ({score:.2f})",
            color=colors[label % len(colors)],
            fontsize=12,
            backgroundcolor="white",
        )

    # Show the plot
    plt.axis("off")
    plt.show()
