from SSLProject.methods.base import BaseMethod
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, ShardingStrategy


def wrap_model_at_fsdp(model: BaseMethod, **kwargs):
    """
    Wrapped training modules in Fully Sharded Data Parallel wrapper.
    For work need what could `model` object was have `get_active_modules_name` method whitch return names of 
    modules whithc will be training.  

    **kwargs (optional) containce parameters for Fully Sharded Data Parallel (see https://docs.pytorch.org/docs/stable/fsdp.html)
    Warning! If model have attribute buffer when his wil bee wrapped with `NO_SHARD` sharding strategy.
    """


    modules_list = model.get_active_modules_name()
    for attr_name in modules_list:
        wrapped_module = FSDP(getattr(model, attr_name), **kwargs)
        setattr(model, attr_name, wrapped_module)

    if hasattr(model, "buffer"):
        kwargs["sharding_strategy"] = ShardingStrategy.NO_SHARD
        wrapped_module = FSDP(getattr(model, "buffer"), **kwargs)
        setattr(model, "buffer", wrapped_module)

    return model
        