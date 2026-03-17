import inspect

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


def cross_entropy(themperature: float=0.1):
    def _inner(inputs: torch.Tensor, target: torch.Tensor):
        teacher_probs = torch.nn.functional.softmax(target / themperature, dim=1)
        student_log_probs = torch.nn.functional.log_softmax(inputs / themperature, dim=1)
        loss = -(teacher_probs * student_log_probs).sum(dim=1).mean() * (themperature ** 2)
        return loss
    return _inner




def change_optimizer_and_sheduler(model, optim, sheduler):
    optim_param = optim.__dict__["defaults"]
    sig = inspect.signature(type(optim).__init__)
    kk = list(sig.parameters.keys())[2::] # remove `self` and `parameters` param
    params = [(k, optim_param[k]) for k in kk if k in optim_param]
    params = dict(params)
    new_optim = type(optim)(model.parameters(), **params)

    if sheduler is not None:
        sig = inspect.signature(type(sheduler).__init__)
        params = list(sig.parameters.keys())
        for to_rem in ["self", "optimizer", "initial_lr", "last_epoch"]:
            if to_rem in params:
                params.remove(to_rem)
        param = {k: sheduler.__dict__[k] for k in params}
        sheduler = type(sheduler)(optimizer=new_optim, **param)
    
    return new_optim, sheduler