import torch


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Extracts off-diagonal elements.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            torch.Tensor:
                flattened off-diagonal elements.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_gradients = torch.stack(grads)
            torch.distributed.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)