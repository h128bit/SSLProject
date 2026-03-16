from typing import Callable, Tuple
import copy

import torch 

from SSLProject.utils.utils import cross_entropy



class BaseMethod(torch.nn.Module):
    """
    The forward method fo the Model must return vector and the model must have public field out_features equal size of output vector! 
    Model must return dict with field `out` containce is result of forward method
    """


    def __init__(self, 
                 model: torch.nn.Module, 
                 loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]|None=None, 
                 device: str|None=None,
                 T: float=0.1):
        super().__init__()

        self.student = model

        self.loss_func = loss_func if loss_func else cross_entropy(T)
        
        match device:
            case None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            case _:
                self.device = device

        self.model_out_feature = self.student.out_features


        self._active_modules = ["student"]


    def get_active_modules_name(self):
        return self._active_modules


    def forward(self, x) -> dict:
        raise NotImplementedError("Not implemented `forward` method in subclass BaseMethod!")
    
    def train_step(self, batch) -> dict:
        raise NotImplementedError("Not implemented `train step` method in subclass BaseMethod!")
    

class BaseMomentum(BaseMethod):

    def __init__(self, 
                 model: torch.nn.Module, 
                 loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]|None=None, 
                 theta: float=0.98, 
                 device: str|None=None,
                 T: float=0.1):
        super().__init__(model, loss_func, device, T)

        self.theta = theta

        # Exponential Moving Average (EMA)
        self.ema_avg_func = lambda avg_model_param, model_param: self.theta * avg_model_param + (1 - self.theta) * model_param
        self.teacher = copy.deepcopy(self.student) #  torch.optim.swa_utils.AveragedModel(self.student, avg_fn=self.ema_avg_func)

        self._active_modules.append("teacher")

        
    def update_teacher_weights(self) -> None:
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data = self.ema_avg_func(param_t, param_s)  # param_t.data * self.theta + param_s.data * (1.0 - self.theta)
            
            for buf_s, buf_t in zip(self.student.buffers(), self.teacher.buffers()):
                buf_t.data = self.ema_avg_func(buf_t, buf_s) #   buf_t.data * self.theta + buf_s.data * (1.0 - self.theta)
    

    def forward(self, x) -> dict:
        return self.train_step(x)
    

    def train_step(self, batch) -> dict:
        view1 = batch[0]
        view2 = batch[1]

        st_out = self.student(view1)
        with torch.no_grad():
            teach_out = self.teacher(view2)

        loss = self.loss_func(st_out, teach_out)

        return {
            "student_out": st_out,
            "teacher_out": teach_out,
            "loss": loss
        }



