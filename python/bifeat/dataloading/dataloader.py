import torch


class SeedGenerator(object):

    def __init__(self,
                 data: torch.Tensor,
                 batch_size: int,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 device="cuda"):
        self.data = data.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            indexes = torch.randperm(self.data.shape[0],
                                     device=self.data.device)
            self.data = self.data[indexes]

        self.step = 0
        if self.drop_last:
            self.last_step = int((self.data.shape[0]) / self.batch_size)
        else:
            self.last_step = int(
                (self.data.shape[0] + self.batch_size - 1) / self.batch_size)

        return self

    def __next__(self):
        if self.step >= self.last_step:
            raise StopIteration

        ret = self.data[self.step * self.batch_size:(self.step + 1) *
                        self.batch_size]
        self.step += 1

        return ret

    def __len__(self):
        return self.last_step

    def is_finished(self):
        return self.step >= self.last_step
