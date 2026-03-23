from typing import Callable

from SSLProject.methods.base import BaseMomentum
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        

class FSDPPrepare:
    def __init__(self):
        pass 


    @staticmethod
    def prepare(method: BaseMomentum,  
                optimizer: Callable, 
                optim_param: dict|None=None,
                sheduler: Callable|None=None,
                sheduler_param: dict|None=None,
                wrap_policy: dict|None=None,
                wrap_teacher: bool=True):
        
        wrap_policy = wrap_policy if wrap_policy  else {}
        optim_param = optim_param if optim_param  else {}
        sheduler_param = sheduler_param if sheduler_param  else {}

        method.student = FSDP(method.student, **wrap_policy, use_orig_params=True)
        method.teacher = FSDP(method.teacher, **wrap_policy, use_orig_params=True) if wrap_teacher else method.teacher

        optimizer = optimizer(method.student.parameters(), **optim_param)
        if sheduler:
            sheduler = sheduler(optimizer, **sheduler_param)

        return method, optimizer, sheduler