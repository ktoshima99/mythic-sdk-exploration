import logging
import queue
import weakref
from collections.abc import Callable, Iterator
from threading import Thread
from typing import Any

from numpy.typing import NDArray
import numpy as np

from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import InputData
from vnnort.data.utils import batch_model_input_data

logger = logging.getLogger(__name__)

QUEUE_SIZE_PER_WORKER = 8  # Number of prefetched samples per worker, before waiting on host to empty queue


class DataloaderWorker(Thread):
    """Helper class implementing a single worker, which can be used to parallelize dataloading.

    All workers share the same input and output queue. After the worker has started as its own thread,
    it pulls the next item from input queue. If no items are available, the worker assumes it is finished.
    Otherwise, the item is fed into `func`, which accepts a single input and returns the result, which is put into the
    output queue. In case the output queue is full the worker waits until there is space available.
    """

    def __init__(self, input_queue: queue.Queue[Any], output_queue: queue.Queue[Any], func: Callable[[Any], Any]):
        """Initialize a worker thread.

        Args:
            input_queue (queue.Queue[Any]): Input queue to load func arguments from
            output_queue (queue.Queue[Any]): Output queue to put func results into
            func (Callable[[Any], Any]): Function to be called by worker thread
        """
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.func = func

    def run(self) -> None:
        """Run the worker loop until the input queue is empty."""
        try:
            while True:
                entry = self.input_queue.get(block=False, timeout=None)  # Raises Empty exception if queue is empty
                result = self.func(entry)
                self.output_queue.put(result)  # Waits if the output queue is full

        except queue.Empty:  # Signal that worker is done
            return
        except Exception:
            raise  # Reraise all others in case something goes wrong


class Dataloader:
    """
    A dataloader for vnnort inference.

    This class is used to create an iterator over the dataset and preprocess the data if a preprocess_func is provided.
    """

    def __init__(  # noqa C901
        self,
        dataset: DatasetBase,
        preprocess_func: Callable[[InputData], Any] | None = None,
        max_samples: int | None = None,
        batch_size: int = 0,
        n_workers: int = 0,
        shuffle: bool = False,
    ) -> None:
        """Initialize the dataloader.

        Args:
            dataset (DatasetBase): The dataset to iterate over.
            preprocess_func (Callable[[InputData], Any] | None): The preprocessing function to apply to the data.
                Defaults to None.
            max_samples (int | None): The maximum number of samples to iterate over.
                If None, all samples are iterated over. If the length of the dataset is less than max_samples,
                it is set to the length of the dataset. Defaults to None.
            batch_size (int): The batch size to use. Defaults to 0. If 0, no batching is done. Otherwise, all samples
                are returned in batches of size batch_size.
            n_workers (int): The number of workers to use for multiprocessing. Defaults to 0. If 0, no multiprocessing
                is done. Otherwise a pool of n_workers processes is created.
            shuffle (bool): Whether to randomize data sampling. Defaults to False.

        Raises:
            ValueError: If dataset is not of type DatasetBase, preprocess_func is not callable,
                or max_samples is not of type int or None. Or if n_workers is not an integer >= 0.
        """
        if not isinstance(dataset, DatasetBase):
            raise ValueError("dataset must be of type DatasetBase")
        if preprocess_func is not None and not callable(preprocess_func):
            raise ValueError("preprocess_func must be a callable")
        if max_samples is not None and not isinstance(max_samples, int):
            raise ValueError("max_samples must be of type int or None")
        if not isinstance(batch_size, int) or batch_size < 0:
            raise ValueError("batch_size needs to be an integer >= 0")
        if not isinstance(n_workers, int) or n_workers < 0:
            raise ValueError("n_workers needs to be an integer >= 0")

        # cap max_samples with datatset size
        if max_samples is None:
            max_samples = len(dataset)
        elif max_samples > len(dataset):
            logger.info(
                f"Max samples {max_samples} is greater than dataset length {len(dataset)}. Using dataset length instead."
            )
            max_samples = len(dataset)

        self.dataset = dataset
        self.preprocess_func = preprocess_func
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.shuffle = shuffle

        # Make sure the pool is cleaned up, when the dataloader is garbage collected
        weakref.finalize(self, self.shutdown)

    def _make_sample_indices(self) -> NDArray[np.int64]:
        if self.shuffle:
            return np.random.choice(len(self.dataset), size=self.max_samples, replace=False)
        return np.arange(self.max_samples, dtype=np.int64)

    def _initialize_workers(self, sample_indices: NDArray[np.int64]) -> None:
        self.input_queue: queue.Queue[int] = queue.Queue()
        for entry in sample_indices:
            self.input_queue.put(int(entry))
        self.output_queue: queue.Queue[Any] = queue.Queue(maxsize=self.n_workers * QUEUE_SIZE_PER_WORKER)
        self.workers = [
            DataloaderWorker(self.input_queue, self.output_queue, self._load_item) for _ in range(self.n_workers)
        ]
        # Start workers
        for worker in self.workers:
            worker.start()

    def shutdown(self) -> None:
        """Shut down all workers."""
        # Nothing to do if single threaded
        if self.n_workers == 0:
            return

        logger.debug("Shutting down dataloader workers.")
        try:
            # Clear input and output queue to stop workers
            logger.debug("Clearing input queue.")
            while not self.input_queue.empty():
                self.input_queue.get()

            logger.debug("Clearing output queue.")
            while not self.output_queue.empty():
                self.output_queue.get()

            logger.debug("Waiting for worker threads.")
            for worker in self.workers:
                if worker.is_alive():
                    # They should be stopped by now. But just in case wait short time.
                    # If they are still alive by then, something is wrong and an exception will be thrown.
                    worker.join(timeout=0.1)

        except queue.Empty:
            return  # After the last item is cleared and Empty exception may be thrown
        except Exception:
            raise

    def _initialize_generator(
        self,
    ) -> Iterator[tuple[InputData, Any]]:
        # Initialize the data generator (either multiprocessing or non-multiprocessing)
        # generate sample_indices
        sample_indices = self._make_sample_indices()

        if self.n_workers > 0:
            # Initialize pool here, to be reused between call to __iter__
            self._initialize_workers(sample_indices)

            def multiprocessing_generator() -> Iterator[tuple[InputData, Any]]:

                try:
                    for _ in range(len(sample_indices)):
                        entry = self.output_queue.get()
                        yield entry
                # Make sure the process pool exits gracefully if users press ctrl-c
                except KeyboardInterrupt:
                    self.shutdown()
                    raise  # Reraise to make sure the rest of th application can handle the exception as wells

            generator = multiprocessing_generator()
        else:

            def non_multiprocessing_generator() -> Iterator[tuple[InputData, Any]]:
                for i in sample_indices:
                    data = self._load_item(int(i))
                    yield data

            generator = non_multiprocessing_generator()
        return generator

    def __iter__(self) -> Iterator[tuple[InputData | list[InputData], Any]]:
        """Iterate over the dataset.

        Yields:
            tuple[InputData | list[InputData], Any]: The next data batch in the dataset. If preprocess_func is provided, it is applied to the data
                and returned instead. In case of batch_size=0, single instances of InputData are returned. In case of
                batch_size>0, lists of InputData are returned.
        """
        generator = self._initialize_generator()
        # Iterate over all entries and yield batches or single entries
        input_data_batch: list[InputData] = []
        model_input_batch: list[InputData] = []
        for data_input, model_input in generator:
            if self.batch_size == 0:
                yield data_input, model_input
            else:
                input_data_batch.append(data_input)
                model_input_batch.append(model_input)
                if len(input_data_batch) == self.batch_size:
                    if self.preprocess_func is not None:
                        yield input_data_batch, batch_model_input_data(model_input_batch)  # type: ignore
                    else:
                        yield input_data_batch, None

                    input_data_batch, model_input_batch = [], []

        # Return the last (possible not fully filled) batch
        if len(input_data_batch) > 0:
            if self.preprocess_func is not None:
                yield input_data_batch, batch_model_input_data(model_input_batch)  # type: ignore
            else:
                yield input_data_batch, None

    def _load_item(self, index: int) -> tuple[InputData, Any]:
        """Load a data sample from the dataset and apply preprocess_func if provided.

        Args:
            index (int): The index of the sample to load.

        Returns:
            tuple[InputData, Any]: The loaded data sample.
        """
        try:
            input_data = self.dataset[index]
            model_input = None
            if self.preprocess_func is not None:
                model_input = self.preprocess_func(input_data)
        except KeyboardInterrupt:
            pass  # Quietly stop

        return input_data, model_input

    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            int: The number of batches / samples returned.
        """
        length = min(self.max_samples, len(self.dataset))
        # zero division guard
        if self.batch_size == 0:
            return length
        batches = length // self.batch_size
        if length % self.batch_size > 0:
            # add residual batch
            batches += 1
        return batches
