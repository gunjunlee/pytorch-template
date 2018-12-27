import torch.utils.data
from torch.utils.data.dataloader import default_collate


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def split(self, lengths, shuffle=False):
        if sum(lengths) != self.__len__():
            ValueError("Sum of input lengths does not match the length of the input dataset!")

        from torch.utils.data import Subset

        def accumulate(iterable):
            it = iter(iterable)
            total = 0
            for element in it:
                total += element
                yield(total)

        indices = [i for i in range(sum(lengths))]

        if shuffle:
            import random
            random.shuffle(indices)

        return [Subset(self, indices[offset-length: offset]) for offset, length in zip(accumulate(lengths), lengths)]


class BaseDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        """Base dataloader

        Parameters
        ----------
        dataset : BaseDataset
        batch_size : int
        shuffle: bool
        sampler: torch.utils.data.sampler
        batch_sampler: torch.utils.data.
        num_workers: int
        collate_fn: function
        pin_memory: bool
        drop_last: bool
        timeout: 0
        worker_init_fn: function
        """

        self.dataset = dataset
        self.kwargs = kwargs
        super(BaseDataLoader, self).__init__(dataset=dataset, **kwargs)

    def split(self, lengths, shuffle=False):
        """split dataloader

        Parameters
        ----------
        lengths : iterable, [int, int, ...]
            lengths of sub-dataloaders
        shuffle : bool, optional
            shuffle indices of data (the default is False, which do not shuffle indices)

        Returns
        -------
        iterable [BaseDataLoader, BaseDataLoader, ...]
            return splited dataloaders
            they share one dataset
        """

        return [BaseDataLoader(dataset=subset, **self.kwargs) for subset in self.dataset.split(lengths, shuffle=shuffle)]


class CustomDataset(BaseDataset):
    """recommended custom dataset structure
    """

    def __init__(self, 
                 dir_root='dataset/train',
                 mode='train',
                 transform=None):
        super(CustomDataset, self).__init__()

        self.dir_root = dir_root
        self.mode = mode
        self.transform = transform

        self.file_list = []

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        # open image
        image = None

        # get name
        name = None

        # pre
        # image = np.array()...

        # transform
        # if self.mode == 'train' and self.transform is not None:
        #     image = self.transform(image)...

        # to tensor
        # image = transforms.functional.to_tensor(image)
        # image = transforms.functional.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        # return image, name...
