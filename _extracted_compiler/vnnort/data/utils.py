import numpy as np
from numpy.typing import NDArray


def batch_model_input_data(data: list[dict[str, NDArray[np.float32]]]) -> dict[str, NDArray[np.float32]]:
    """Batch model input data.

    This function takes in a list of possible model inputs as they are returned by any models' preprocess function
    and returns a single dictionary, which can be fed into the model instead with stackable inputs (like e.g.
    images) being stacked in the batch dimension. Inputs without a batch dimension are not stacked.

    Args:
        data (list[dict[str, NDArray[np.float32]]]): List of model inputs.
    Raises:
        TypeError: If data is not a list or a list of dicts of numpy arrays
        RuntimeError: If data shape changes between entries and non stackable entries differ

    Returns:
        dict[str, NDArray[np.float32]]: Batched model inputs
    """
    if not isinstance(data, list):
        raise TypeError("Data needs to be a list")
    if len(data) == 0:
        raise TypeError("Data needs to be non-empty")
    if not all(isinstance(entry, dict) for entry in data):
        raise TypeError("Data needs to be a list of non-empty dicts")
    if not all([len(entry) > 0 and all(isinstance(value, np.ndarray) for value in entry.values()) for entry in data]):
        raise TypeError("Data needs to be a list of dicts of numpy arrays")
    if not all(
        all(key in data[0] and data[0][key].shape == entry[key].shape for key in entry.keys()) for entry in data
    ):
        raise TypeError("Dictionaries need to contain entries with same shape")

    result = {}
    for key in data[0].keys():
        entries = [entry[key] for entry in data]
        # Only stack if there is a batch dimension with size 1
        if entries[0].ndim > 1 and entries[0].shape[0] == 1:
            result[key] = np.concatenate(entries, axis=0)
        # If there is no batch dimension, take the first entry and expect that all other entries in list are the same
        elif entries[0].ndim <= 1 and (len(entries) == 1 or all(np.allclose(entries[0], e) for e in entries[1:])):
            result[key] = entries[0]
        elif len(entries) == 1:
            result[key] = entries[0]
        else:
            raise RuntimeError("Unexpected input shape")
    return result


def unbatch_model_output_data(data: dict[str, NDArray[np.float32]]) -> list[dict[str, NDArray[np.float32]]]:
    """Unbatch model output data.

    This function takes in a dictionary of possible model outputs as they are returned by any model
    and returns a list of dictionaries, where each dictionary contains the outputs for one sample in the
    batch.

    Args:
        data (dict[str, NDArray[np.float32]]): Dictionary of model outputs.

    Returns:
        list[dict[str, NDArray[np.float32]]]: List of dictionaries, one entry per batch dimension.

    Raises:
        TypeError: If data is not a dict of numpy arrays
        RuntimeError: If data shape changes between entries and non stackable entries differ

    Examples:
        >>> data = {'logits': np.array([[1, 2, 3], [4, 5, 6]]), 'features': np.array([[0.1, 0.2], [0.3, 0.4]])}
        >>> unbatch_model_output_data(data)
        [{'logits': array([[1, 2, 3]]), 'features': array([[0.1, 0.2]])}, {'logits': array([[4, 5, 6]]), 'features': array([[0.3, 0.4]])}]
    """
    if not isinstance(data, dict):
        raise TypeError("Data needs to be a dict")

    if len(data) == 0:
        raise TypeError("Data needs to be non-empty")

    if not all(isinstance(value, np.ndarray) for value in data.values()):
        raise TypeError("Data needs to be a dict of numpy arrays")

    # Go over each entry and extract a possible batch size (if applicable)
    possible_batch_sizes = {}
    for key, entry in data.items():
        if entry.ndim > 1 and entry.shape[0] >= 1:
            possible_batch_sizes[key] = entry.shape[0]
        else:
            raise RuntimeError("Unexpected output shape. Needs to contain a batch dimension")

    # All entries must match
    if len(set(possible_batch_sizes.values())) > 1:
        raise RuntimeError("Batch sizes differ between entries")

    # Collect result list
    results = []
    batch_size = next(iter(data.values())).shape[0]
    for i in range(batch_size):
        result = {}
        for key, values in data.items():
            result[key] = values[i][None, ...]
        results.append(result)

    return results
