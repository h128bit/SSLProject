from typing import Callable, Tuple
import copy

import torch 

from SSLProject.utils.utils import cross_entropy



class BaseMomentum(torch.nn.Module):
    """
    The forward method fo the Model must return vector and the model must have public field out_features equal size of output vector! 
    Model must return dict with field `out` containce is result of forward method
    """

    def __init__(
            self, 
            student: torch.nn.Module, 
            teacher: torch.nn.Module|None=None, 
            loss_func: Callable|None=None, 
            T: float=0.1, 
            theta: float=0.98):
        
        super().__init__()
        
        self.student = student

        if teacher is None:
            self.teacher = copy.deepcopy(student)
        else:
            self.teacher = teacher 

        self.theta = theta

        self.loss_func = loss_func if loss_func else cross_entropy(T)
        self.ema_avg_func = lambda avg_model_param, model_param: self.theta * avg_model_param + (1 - self.theta) * model_param


    def update_teacher_weights(self) -> None:
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data = self.ema_avg_func(param_t, param_s)  
            
            # for buf_s, buf_t in zip(self.student.buffers(), self.teacher.buffers()):
            #     buf_t.data = self.ema_avg_func(buf_t, buf_s) 
    
 
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