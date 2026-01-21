import torch
from torch import nn
from SSLProject.utils.utils import gather


class SupportBuffer(nn.Module):
    def __init__(self, 
                 buffer_capacity: int=98304, 
                 vectors_dim: int=256):
        super().__init__()

        self.register_buffer("batch_capacity_", torch.tensor(buffer_capacity, dtype=torch.int32)) # scalar
        self.register_buffer("vectors_dim_", torch.tensor(vectors_dim, dtype=torch.int32)) # scalar
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.int32)) # scalar
        self.register_buffer("buffer", nn.functional.normalize(torch.rand(buffer_capacity, vectors_dim), dim=0)) 
        # self.register_buffer("buffer_idx", -1 * torch.ones(buffer_capacity, dtype=torch.int32))


    @property
    def buffer_capacity(self) -> int:
        return self.batch_capacity_.item()


    @property
    def vectors_dim(self) -> int:
        return self.vectors_dim_.item()


    @torch.no_grad()
    def put(self, 
            batch: torch.Tensor) -> None:
        batch = gather(batch)

        shift = batch.shape[0]

        assert shift <= self.buffer_capacity, f"batch size must be equal or less than buffer capacity: batch size {shift}, capacity {self.buffer_capacity}"
        assert batch.shape[1] == self.vectors_dim, f"Not correct vector dimension in batch: vector dimension {batch.shape[1]}, but must be equal {self.vectors_dim}"

        ptr = int(self.ptr)

        if self.ptr + shift <= self.buffer_capacity:
            self.buffer[ptr:ptr + shift] = batch
        else:
            tail = self.buffer_capacity - ptr
            self.buffer[ptr::] = batch[0:tail]
            self.buffer[0:shift - tail] = batch[tail::]

        self.ptr.fill_((ptr + shift) % self.buffer_capacity) 




class SupportBufferKNN(SupportBuffer):
    def __init__(self, 
                 buffer_capacity: int=98304, 
                 vectors_dim: int=256, 
                 k: int=5):
        super().__init__(buffer_capacity, vectors_dim)
        self.k = k


    @torch.no_grad()
    def put(self,
            batch: torch.Tensor) -> None:
        batch = nn.functional.normalize(batch, dim=1)
        super().put(batch)


    @torch.no_grad()
    def find_nn(self, 
                z: torch.Tensor,
                k: int|None=None) -> torch.Tensor:
        k = k if k else self.k 

        z = nn.functional.normalize(z, dim=1)
        sims = z @ self.buffer.T

        _, idx = sims.topk(k, dim=1)
        neighbors = self.buffer[idx]

        return neighbors

