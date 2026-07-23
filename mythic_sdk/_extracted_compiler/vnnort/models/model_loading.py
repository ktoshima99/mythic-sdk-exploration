from __future__ import annotations

from typing import Dict, Type

from vnnort.models.vid_model import VidModel

# List of models that support the FULL flow
supported_models = [
    "image_classification.huggingface.google.vit_base_patch16.ViTBasePatch16",
    "image_classification.mlperf.resnet50.Resnet50",
    "image_classification.torchvision.alexnet.AlexNet",
    "image_classification.torchvision.convnext_base.ConvNextBase",
    "image_classification.torchvision.convnext_tiny.ConvNextTiny",
    "image_classification.torchvision.efficientnet_b0.EfficientNetB0",
    "image_classification.torchvision.efficientnet_b7.EfficientNetB7",
    "image_classification.torchvision.efficientnet_v2_m.EfficientNetV2M",
    "image_classification.torchvision.googlenet.GoogleNet",
    "image_classification.torchvision.inception_v3.InceptionV3",
    "image_classification.torchvision.mnasnet0_5.MNASNet0_5",
    "image_classification.torchvision.mobilenet_v2.MobileNetV2",
    "image_classification.torchvision.mobilenet_v3_large.MobileNetV3Large",
    "image_classification.torchvision.regnet_x_1_6gf.Regnet_x_1_6gf",
    "image_classification.torchvision.resnet18.Resnet18",
    "image_classification.torchvision.resnet18_softmax.Resnet18Softmax",
    "image_classification.torchvision.resnet34.Resnet34",
    "image_classification.torchvision.resnet50.Resnet50",
    "image_classification.torchvision.resnet101.Resnet101",
    "image_classification.torchvision.resnext50_32x4d.ResNext50_32x4d",
    "image_classification.torchvision.shufflenet_v2_x2_0.ShuffleNet_v2_x2_0",
    "image_classification.torchvision.squeezenet1_0.SqueezeNet1_0",
    "image_classification.torchvision.squeezenet1_1.SqueezeNet1_1",
    "image_classification.torchvision.vgg11_bn.VGG11_BN",
    "image_classification.torchvision.wide_resnet50_2.Wide_Resnet50_2",
    "customers.bosch.convnext.ConvNext",
    "customers.bosch.fasterrcnn.FasterRCNN",
    "image_detection.detr.DETR",
    "image_detection.mlperf.mobilenetssd.MobilenetSSD",
    "image_detection.mlperf.retinanet.RetinaNet",
    "image_detection.yolo.yolov4.YoloV4",
    "image_detection.yolo.yolov5m.YoloV5m",
    "image_detection.yolo.yolov5n.YoloV5n",
    "image_detection.yolo.yolov5s.YoloV5s",
    "image_detection.yolo.yolov7.YoloV7",
    "image_detection.yolo.yolov8l.YoloV8l",
    "image_detection.yolo.yolov8m.YoloV8m",
    "image_detection.yolo.yolov8n.YoloV8n",
    "image_detection.yolo.yolov8s.YoloV8s",
    "image_segmentation.tensorflow.deeplabv3plus.DeepLabv3Plus",
    "question_answering.mlperf.bert.MLPerfQABert",
    "text_generation.huggingface.bert.qa_bert.QABertBase",
    "text_generation.huggingface.bert.qa_bert.QABertLarge",
    "visual_question_answering.qwen2_5vl.qwen2_5vl.Qwen2_5_VL_3B_Instruct",
    "visual_question_answering.qwen2_5vl.qwen2_5vl.Qwen2_5_VL_7B_Instruct",
    "text_generation.huggingface.llama.llama3_2_1B.LLama3_2_1B_Prefill",
    "text_generation.huggingface.llama.llama3_2_1B.LLama3_2_1B_Decode",
    "text_generation.huggingface.llama.llama3_2_1B.LLama3_2_1B",
    "image_classification.timm.repvgg.repvgg.RepVGGA0",
    "customers.bosch.bosch_evo50.BoschEvo50",
    "customers.mythic.yolov8pose_postprocessing.MythicYoloV8PosePostprocessing",
    "customers.bosch.prep_bevmodel.BoschPrepBEVModel",
    "customers.mythic.bevformer.bevformer_tiny.BevformerTiny",
]


def load_model_class(model_name: str) -> Type[VidModel]:
    """Dynamically loads the class of a model that implements the VidModel base class, given the name of that class.

    Args:
        model_name (str): The name of the model class to load.

    Returns:
        Type[VidModel]: The model class corresponding to the provided name.

    Raises:
        ValueError: If the specified model name is not available.
    """
    available_model_classes = get_available_model_classes()
    if model_name in available_model_classes and model_name in supported_models:
        ModelClass: Type[VidModel] = available_model_classes[model_name]
        return ModelClass
    else:
        available_models_str = "\n".join(available_model_classes.keys())
        raise ValueError(f"{model_name} is currently not supported. \nChoose one of:\n{available_models_str}")


def get_available_model_classes() -> Dict[str, Type[VidModel]]:
    """Retrieve all available model classes that inherit from VidModel.

    This function recursively finds all subclasses of VidModel, constructs a dictionary
    with the model name as the key, and the model class as the value.

    Returns:
        Dict[str, Type[VidModel]]: A dictionary where keys are model names and values are model classes.
    """

    def _get_all_subclasses(cls: type[VidModel]) -> set[type[VidModel]]:
        """Recursively finds all subclasses of a given class.

        Args:
            cls (type[VidModel]): The base class to search for subclasses.

        Returns:
            set[type[VidModel]]: A set of all subclasses derived from the given base class.
        """
        subclasses = set(cls.__subclasses__())
        for subclass in cls.__subclasses__():
            subclasses.update(_get_all_subclasses(subclass))
        return subclasses

    available_model_classes = _get_all_subclasses(VidModel)  # type: ignore # Todo: fix this
    available_model_classes_ret = {
        ".".join((ModelClass.__module__ + "." + ModelClass.__name__).split(".")[3:]): ModelClass
        for ModelClass in available_model_classes
    }

    # Only add modules, which are part of the model zoo
    available_model_classes_ret = {
        m: Class
        for m, Class in available_model_classes_ret.items()
        if "model_zoo" in Class.__module__ and m in supported_models
    }

    return available_model_classes_ret
