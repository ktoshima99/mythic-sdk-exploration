from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Generator, Mapping

import onnx

from vnnort.inference.runtime.onnx_runtime import ONNXRuntime
from vnnort.utils.onnx_utils.onnx_hooks import OnnxHookBase


class HookedOnnxInferenceSession:
    """Attach custom hooks to ONNX model tensors and fetch intermediate results.

    This is implemented by marking intermediate tensors as model outputs. When calling
    the ONNX Runtime InferenceSession, these outputs are returned. Internally, they are
    routed to workers that call the corresponding ``hook.on_call()`` method.

    After the last call to ``run()``, the final results can be gathered via ``results()``.

    Example::

        onnx_model = onnx.load("model.onnx")

        hooks = {}
        for node in onnx_model.graph.node:
            for output in node.output:
                hooks[output] = OnnxOutputHook()

        with HookedOnnxInferenceSession.create(onnx_model, hooks) as session:
            input_data = {"input": np.ones([1, 3, 10, 10], dtype=np.float32)}
            session.run(input_data)

            for tensor_name, hook in hooks.items():
                print(tensor_name, hook.compute())

            results = session.results()
    """

    def __init__(self, onnx_model: onnx.ModelProto, hooks: dict[str, OnnxHookBase], n_workers: int = 1):
        """Initialize the hooked inference session for model with the provided hooks.

        This constructor must not be used directly! Use the staticmethod create() with a context manager instead!
        This is necessary to clean up the onnxruntime and spawned processes afterwards.

        Args:
            onnx_model (onnx.ModelProto): Onnx model, where hooks will be attached to tensors.
            hooks (dict[str, OnnxHookBase]): Dictionary with tensor names and corresponding hooks.
            n_workers (int): Number of workers to use for processing calls to hooks.

        Raises:
            RuntimeError: If HookedOnnxInferenceSession has not been created
            ValueError: If onnx_model is not a onnx.ModelProto
        """
        if not hasattr(self, "_from_context_manager"):
            raise RuntimeError("HookedOnnxInferenceSession need to be created with context manager!")
        if not isinstance(onnx_model, onnx.ModelProto):
            raise ValueError("Expected onnx.ModelProto")

        # Keep track of hooks. These are later overwritten by results of workers
        self._hooks = hooks
        self._model = onnx_model

        # Overwrite model outputs, but keep track of original ones
        self._original_model_outputs = deepcopy(onnx_model.graph.output)
        new_model_outputs = [name for name in hooks.keys()]
        self._set_model_outputs(new_model_outputs)

        # Initalize ONNX runtime with modified model
        self._ort_session = ONNXRuntime(self._model)

        # Create worker processes and queues
        if n_workers == 0:
            n_workers = 1  # We need at least one worker, this won't make any difference
        elif n_workers < 0:
            raise ValueError("n_workers must be >= 0")
        self._pool = ThreadPoolExecutor(n_workers)

    @classmethod
    @contextmanager
    def create(
        cls, onnx_model: onnx.ModelProto, hooks: Mapping[str, OnnxHookBase], n_workers: int = 1
    ) -> Generator["HookedOnnxInferenceSession", None, None]:
        """Create an instance of the HookedOnnxInferenceSession to be used in a context manager.

        Args:
            onnx_model (onnx.ModelProto): Onnx model, where hooks will be attached to tensors.
            hooks ( Mapping[str, OnnxHookBase]): Dictionary with tensor names and corresponding hooks.
            n_workers (int): Number of workers to use for processing calls to hooks.

        Yields:
            HookedOnnxInferenceSession: Session instance that can be used within a
                context manager block.

        Returns:
            Generator["HookedOnnxInferenceSession", None, None]: HookedInferenceSession to be used in context manager

        """
        # We need to create the object this way to make sure the init method is not used manually
        try:
            instance = object.__new__(HookedOnnxInferenceSession)
            instance._from_context_manager = True  # type: ignore
            instance.__init__(onnx_model, hooks, n_workers)  # type: ignore
            yield instance
        finally:
            # Perform cleanup
            del instance._ort_session
            instance._pool.shutdown(cancel_futures=True)

            # Reinstate the original model outputs
            # instance._set_model_outputs(instance._original_model_outputs)
            model = instance._model
            model.graph.ClearField("output")
            for output in instance._original_model_outputs:
                model.graph.output.append(output)

    def run(self, input_data: Any) -> None:
        """Run a single inference step for the given input data and call all registred hooks with intermediate data.

        Args:
            input_data (Any): Model input for single inference step

        Returns:
            None: Nothing is returned.
        """
        # Run onnxruntime session with input data
        model_outputs = self._ort_session(input_data)

        # Distribute results between processes
        futures = []
        for index, (name, model_output) in enumerate(model_outputs.items()):
            worker_future = self._pool.submit(self._hooks[name].on_call, model_output)
            futures.append(worker_future)

            # Wait for all futures to finish
            for future in as_completed(futures):
                pass
            # self._hooks[name].on_call(model_output)

    def results(self) -> dict[str, OnnxHookBase]:
        """Get the final results of all hooks.

        Returns:
            dict[str, OnnxHookBase]: Dictionary with tensor names and corresponding hooks
        """
        # Wait for all threads in pool to finish
        self._pool.shutdown(wait=True)
        return self._hooks

    def _set_model_outputs(self, output_names: list[str]) -> None:
        """Set output tensors as model outputs.

        Args:
            output_names (list[str]): Name of tensors to be added as model outputs

        Returns:
            None: Nothing is returned
        """
        model = self._model
        model.graph.ClearField("output")
        for name in output_names:
            model.graph.output.append(onnx.ValueInfoProto(name=name))

    def _get_model_outputs(self) -> list[str]:
        """Get the current list of output tensors from the model.

        Returns:
            list[str]: Names of tensors currently returned by the model
        """
        output_names = []
        for output in self._model.graph.output:
            output_names.append(output.name)
        return output_names
