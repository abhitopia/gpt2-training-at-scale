import time
from torch.multiprocessing import Queue, Process
import numpy as np
from tqdm import tqdm


class DatasetIteratorWorker(Process):
    def __init__(self, dataset, indices, request_queue, result_queue):
        super().__init__()
        self.dataset = dataset
        self.result_queue = result_queue
        self.request_queue = request_queue
        self.indices = indices

    def fill_queue(self, idx, buffersize):
        batch = []
        for i in range(buffersize):
            if idx < len(self.indices):
                sample_idx = self.indices[idx]
                sample = self.dataset[sample_idx]
                batch.append((sample_idx, sample))
            idx += 1
        self.result_queue.put(batch)
        return idx

    def run(self):
        idx = self.fill_queue(0, 10)
        while idx < len(self.indices):
            _ = self.request_queue.get()
            idx = self.fill_queue(idx, 1)

    def terminate(self):
        del self.dataset
        super().terminate()


class DatasetIterator:
    def __init__(self, dataset, num_workers=0, synchronous=True, indices=None):
        self.dataset = dataset
        self.num_workers = num_workers if num_workers >= 0 else 0
        self.synchronous = synchronous
        if indices is not None:
            assert (max(indices) < len(self.dataset) and min(indices) >= 0), "Index Out of range"
            self.indices = indices
        else:
            self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        if self.num_workers == 0:
            for i in self.indices:
                yield self.dataset[i]

            return

        workers = []
        num_samples = len(self.indices)

        result_queue = Queue()
        request_queue = Queue()
        num_indices_per_worker = num_samples // self.num_workers
        remainder = num_samples % self.num_workers

        # indices = np.arange(num_indices_per_worker * self.num_workers).reshape(-1, self.num_workers).T
        indices = np.asarray(self.indices[:num_indices_per_worker*self.num_workers]).reshape(-1, self.num_workers).T

        for idx in range(self.num_workers):
            worker_indices = list(indices[idx])
            if idx < remainder:
                worker_indices += [self.indices[indices.size + idx]]

            worker = DatasetIteratorWorker(indices=worker_indices,
                                           request_queue=request_queue,
                                           result_queue=result_queue,
                                           dataset=self.dataset)
            worker.daemon = True
            worker.start()
            workers.append(worker)

        buffer = {}
        current_idx = 0
        while current_idx < num_samples:
            samples = result_queue.get()
            if self.synchronous:
                for sample_idx, sample in samples:
                    buffer[sample_idx] = sample

                while current_idx < num_samples and self.indices[current_idx] in buffer:
                    next_sample_idx = self.indices[current_idx]
                    yield buffer[next_sample_idx]
                    del buffer[next_sample_idx]
                    current_idx += 1
            else:
                for _, sample in samples:
                    current_idx += 1
                    yield sample

            request_queue.put(None)

        for worker in workers:
            worker.terminate()

    def __len__(self):
        return len(self.indices)

    def __del__(self):
        del self.dataset


class DummyDataset:
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples

    def __getitem__(self, item):
        assert item < self.num_samples
        import numpy as np
        time.sleep(np.random.rand())
        return item

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    indices = [1, 2, 3, 8, 9, 10, 20, 30, 21]
    loader = DatasetIterator(dataset=DummyDataset(num_samples=234), num_workers=5, synchronous=True, indices=indices)

    samples = []
    for sample in tqdm(loader):
        print(sample)
        #samples.append(sample)

    print(len(samples))

