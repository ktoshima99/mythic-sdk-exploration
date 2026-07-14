from copy import deepcopy

import onnx


def make_initializers_unique(model: onnx.ModelProto) -> None:
    """Make all initializers being used by multiple nodes unique.

    Args:
        model (onnx.ModelProto): Model to be modified

    Returns:
        None: This function returns nothing
    """
    # Extract all initializers and track, where they are used as inputs
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    initializer_to_input_mapping: dict[str, list[onnx.NodeProto]] = {i: [] for i in initializers.keys()}
    for node in model.graph.node:
        for inp in node.input:
            if inp in initializer_to_input_mapping:
                initializer_to_input_mapping[inp].append(node)

    # Find all initializers being used more than once
    for initializer_name, nodes in initializer_to_input_mapping.items():
        if len(nodes) > 1:
            # Create seperate initializers by duplicating the original one
            original_initializer = initializers[initializer_name]
            original_name = original_initializer.name
            # Only keep it linked to the first node, the rest gets new ones
            for index, node in enumerate(nodes[1:]):
                # Create new initializer and add to models initializer list
                new_initializer = deepcopy(original_initializer)
                new_initializer.name = original_name + f"_{index}"
                model.graph.initializer.append(new_initializer)

                # Replace input name in node inputs
                for i in range(len(node.input)):
                    if node.input[i] == original_name:
                        node.input[i] = new_initializer.name
                        break  # Ensures that this code also works when initializer is used multiple time WITHIN node
