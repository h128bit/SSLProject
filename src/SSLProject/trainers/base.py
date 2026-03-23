import os
import copy
from typing import Iterable
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType

from SSLProject.utils.enviroment_utils import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm 
else:
    from tqdm import tqdm 

from SSLProject.utils.enviroment_utils import get_logger
from SSLProject.trainers.loggers import get_process_logger
from SSLProject.methods.base import BaseMomentum



class BaseTrainer:
    def __init__(self, 
                 method: BaseMomentum, 
                 optimizer: torch.optim.Optimizer, 
                 num_epoch: int, 
                 dataloader: torch.utils.data.DataLoader,
                 sheduler: torch.optim.lr_scheduler.LRScheduler,
                 update_teacher_after_n_epoch: int=0,
                 update_teacher_each_n_step: int=1,
                 save_model: bool=True,
                 save_model_each_n_epochs: int=1,
                 project_root_or_url: str="",
                 project_name: str="runs",
                 run_name: str="ssl_run",
                 logger: str="simple",
                 accumulate_grad: bool=True,
                 accumulate_step: int=1): 
        self.logger = get_logger("TrainerLogger")

        self.project_root = project_root_or_url
        self.project_name = project_name
        self.run_name = run_name 
        self.process_logger = get_process_logger(logger, self.project_root, self.project_name, self.run_name)
        self.logger.info(f"{type(self.process_logger)} was created")
        
        self.num_epoch = num_epoch

        self.save_model = save_model
        self.start_update = update_teacher_after_n_epoch
        self.update_each_n_step = update_teacher_each_n_step
        self.save_model_each_n_epochs = save_model_each_n_epochs 

        self.accumulate_grad = accumulate_grad
        self.accumulate_step = accumulate_step

        self.dataloader = dataloader

        self.method = method
        self.optimizer = optimizer
        self.sheduler = sheduler

        self.step_history = {}


    def start_train_hook(self):
        self.process_logger.start_experiment()
        self.logger.info("===== Start training =====") 
    
    def start_epoch_hook(self, epoch):
        pass
    
    def train_step(self, batch, do_step) -> dict:
        raise NotImplementedError
    
    def update_teacher_weights(self):
        self.method.update_teacher_weights()

    def save_model_hook(self, model_save_step: int):
        pass
    
    def end_train_step_hook(self):
        pass

    def end_train_hook(self):
        self.process_logger.end_experiment()



    def train(self): 
        self.start_train_hook()
        step = 0
        batches_per_epoch = len(self.dataloader)

        try:
            
            for epoch in tqdm(range(self.num_epoch), desc="epoch: "):
                self.start_epoch_hook(epoch)

                for batch in tqdm(self.dataloader, desc="batch progress: ", total=batches_per_epoch, leave=False):
                    step += 1
                    
                    do_step = (step % self.accumulate_step == 0)

                    loss_dict = self.train_step(batch, do_step)

                    if epoch >= self.start_update and step % self.update_each_n_step == 0:
                        self.update_teacher_weights()
                    
                    self._update_step_history(loss_dict)
                
                if batches_per_epoch % self.accumulate_step != 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.sheduler:
                    self.sheduler.step()

                self._do_log(epoch)

                model_save_step = step//batches_per_epoch
                if self.save_model and (epoch+1) % self.save_model_each_n_epochs == 0:
                    self.save_model_hook(model_save_step)
                
                self.end_train_step_hook()

        finally:
            self.end_train_hook()



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

    
    
        



