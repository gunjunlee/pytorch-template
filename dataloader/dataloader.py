import torch.utils.data
from torch.utils.data.dataloader import default_collate

class Base_dataset(torch.utils.data.Dataset):
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
        
        return [Subset(self, indices[offset-length, offset]) for offset, length in zip(accumulate(lengths), lengths)]

class Base_dataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        super(Base_dataloader, self).__init__(dataset=dataset, **kwargs)

    def split(self, lengths, shuffle=False):
        return [Base_dataloader(dataset=subset, **self.kwargs) for subset in self.dataset.split(lengths, shuffle=shuffle)]