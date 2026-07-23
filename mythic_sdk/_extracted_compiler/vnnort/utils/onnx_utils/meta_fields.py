from onnx import ModelProto


def _set_onnx_meta_field(model: ModelProto, key: str, value: str) -> None:
    """Set the key field in a onnx objects' meta data property to value.

    This function is used to keep track of meta data like e.g. model names and persist them
    with the onnx file.
    """
    # Check if the key already exists
    for meta in model.metadata_props:
        if meta.key == key:
            meta.value = value
            break
    else:
        # Add new metadata if key does not exist
        new_meta = model.metadata_props.add()
        new_meta.key = key
        new_meta.value = value


def _get_onnx_meta_field(model: ModelProto, key: str) -> str | None:
    """Retrieve the value of the field key from the onnx objects' meta data property.

    Args:
        model (ModelProto): Onnx ModelProto object
        key (str): key to retrieve

    Returns:
        str | None: The resulting value or None if it does not exist.
    """
    for meta in model.metadata_props:
        if meta.key == key:
            return str(meta.value)
    return None  # Return None if key is not found
