import numpy as np
import pickle

class ImageDataset():

    def __init__(self, datasets, transform=[], type='train', resize = False):
        super(ImageDataset, self).__init__()

        with open(datasets, mode='rb') as f:
            datasets = pickle.load(f)


        self.images    = datasets['features']
        self.labels    = datasets['labels']
        self.transform = transform
        self.type      = type
        self.resize    = resize
        self.width     = datasets['features'].shape[1]
        self.height    = datasets['features'].shape[2]

    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def get_train_item(self,index):
        image = self.images[index]
        label = self.labels[index]
        for t in self.transform:
            image = t(image)

        return image, label, index

    def get_test_item(self,index):
        image = self.images[index]
        label = self.labels[index]
        for t in self.transform:
            image = t(image)

        return image, index


    def __getitem__(self, index):

        if self.type=='train': return self.get_train_item(index)
        if self.type=='test':  return self.get_test_item (index)

    def __len__(self):
        return len(self.images)


class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, batch_size=1, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.collate_fn = []

        sampler = RandomSampler(dataset)

        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)


class BatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class RandomSampler(object):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.random.permutation(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.sample_iter = iter(self.batch_sampler)


    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = [self.dataset[i] for i in indices]
        batch = np.asarray(batch)
        batch = np.transpose(batch,(1,0))

        return batch

    def __iter__(self):
        return self


    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")


if __name__ == "__main__":

    dataset = ImageDataset(datasets="data/train.p")
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        )

    for i, (images,labels,indices) in enumerate(train_loader, 0):
        print('i=%d: ' % (i))