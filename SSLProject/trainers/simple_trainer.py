import torch 
from typing import Iterable
import logging


from SSLProject.utils.enviroment_utils import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm 
else:
    from tqdm import tqdm 

from SSLProject.utils.enviroment_utils import get_logger
from SSLProject.trainers.loggers import SimpleLogger, SimpleMLFlowLogger
from SSLProject.methods.base import BaseMethod


class SimpleTrainer:
    def __init__(self, 
                 method: BaseMethod, 
                 optimizer: torch.optim.Optimizer, 
                 num_epoch: int, 
                 dataloader: Iterable,
                 sheduler: torch.optim.lr_scheduler.LRScheduler,
                 update_teacher_after_n_epoch: int=0,
                 update_teacher_each_n_step: int=1,
                 save_model: bool=True,
                 save_model_each_n_epochs: int=1,
                 project_root_or_url: str="",
                 project_name: str="runs",
                 run_name: str="ssl_run",
                 logger: str="simple"):
        
        self.logger = get_logger("SimpleTrainer")

        self.method = method 

        self.optimizer = optimizer    
        self.sheduler = sheduler 
        self.dataloader = dataloader

        self.num_epoch = num_epoch

        self.start_update = update_teacher_after_n_epoch
        self.update_each_n_step = update_teacher_each_n_step

        self.save_model = save_model
        self.save_model_each_n_epochs = save_model_each_n_epochs

        match logger:
            case "simple":
                self.process_logger = SimpleLogger(root=project_root_or_url,
                                           project_name=project_name,
                                           run_name=run_name) 
                self.logger.info("SimpleLogger was created")
            case "mlflow":
                self.process_logger = SimpleMLFlowLogger(url=project_root_or_url,
                                                 project_name=project_name,
                                                 run_name=run_name)
                self.logger.info("SimpleMLFlowLogger was created")
            
            case _:
                raise ValueError(f"Unknow logger {logger}. Supported 'simple' or 'mlflow'")

        self.step_history = {}


    def train(self): 
        self.process_logger.start_experiment()
        self.logger.info("=== Start training ===") 
        step = 0
        batch_size = len(self.dataloader)

        try:
            for epoch in tqdm(range(self.num_epoch), desc="epoch: "):
                for batch in tqdm(self.dataloader, desc="batch progress: ", total=batch_size, leave=False):
                    step += 1
                    self.optimizer.zero_grad()

                    loss_dict = self.method(batch)
                    loss = loss_dict["loss"]

                    loss.backward()

                    if epoch >= self.start_update and step % self.update_each_n_step == 0:
                        self.method.update_teacher_weights()
                    
                    self._update_step_history(loss_dict)
                
                if self.sheduler:
                    self.sheduler.step()

                self._do_log(epoch)

                if self.save_model and step % self.save_model_each_n_epochs == 0:
                    self.process_logger.save_model_state_dict(self.method.student, "student_model", step//batch_size)
                    self.process_logger.save_model_state_dict(self.method.teacher.module, "teacher_model", step//batch_size)
        finally:
            self.process_logger.end_experiment()


    def _update_step_history(self, d: dict):
        for k, v in d.items():
            v = v.item()
            if k in self.step_history:
                    self.step_history[k].append(v)
            else:
                self.step_history[k] = [v]


    def _do_log(self,
                epoch: int):
        for k, v in self.step_history.items():
            v = sum(v)/len(v)
            self.step_history[k] = v
        self.process_logger.loglog(self.step_history, epoch)
        self.step_history = {}


