from typing import Callable, Tuple

import torch 

from SSLProject.utils.utils import cross_entropy



class BaseMethod(torch.nn.Module):
    """
    The forward method fo the Model must return vector and the model must have public field out_features equal size of output vector!  
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


    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Not implemented `forward` method in subclass BaseMethod!")
    
    def train_step(self, batch) -> dict[str, torch.Tensor]:
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
        self.ema_avg_func = lambda avg_model_param, model_param, num_averaged: self.theta * avg_model_param + (1 - self.theta) * model_param
        self.teacher = torch.optim.swa_utils.AveragedModel(self.student, avg_fn=self.ema_avg_func)

        
    def update_teacher_weights(self) -> None:
        self.teacher.update_parameters(self.student)
    

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        First view for student, second to teacher
        """

        view1 = x[0]
        view2 = x[1]

        st_out = self.student(view1)
        teach_out = self.teacher(view2)

        return st_out, teach_out
    

    def train_step(self, batch) -> dict[str, torch.Tensor]:
        out = self.forward(batch)


        loss = self.loss_func(*out)

        return {
            "z_student": out[0],
            "z_teacher": out[1],
            "loss": loss
        }
