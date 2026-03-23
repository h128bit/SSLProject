import os
import copy
from typing import Iterable, Callable
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
from SSLProject.data_parallel.shared_data_parallel import FSDPPrepare
from SSLProject.methods.base import BaseMomentum
from SSLProject.trainers.base import BaseTrainer
from SSLProject.trainers.loggers import get_process_logger



class FSDPTrainer(BaseTrainer):
    def __init__(self, 
                 method: BaseMomentum, 
                 optimizer: Callable, 
                 num_epoch: int, 
                 dataloader: torch.utils.data.DataLoader,
                 sheduler: Callable|None=None,
                 optim_param: dict|None=None,
                 sheduler_param: dict|None=None,
                 update_teacher_after_n_epoch: int=0,
                 update_teacher_each_n_step: int=1,
                 save_model: bool=True,
                 save_model_each_n_epochs: int=1,
                 project_root_or_url: str="",
                 project_name: str="runs",
                 run_name: str="ssl_run",
                 logger: str="simple",
                 accumulate_grad: bool=True,
                 accumulate_step: int=1
                 ):

        self.is_main_rank = dist.get_rank() == 0

        if self.is_main_rank:
            with torch.no_grad():
                self.model_copy = copy.deepcopy(method.student).cpu().eval()

        method, optimizer, sheduler = FSDPPrepare.prepare(method=method, 
                                                          optimizer=optimizer, 
                                                          optim_param=optim_param, 
                                                          sheduler=sheduler, 
                                                          sheduler_param=sheduler_param)
        self.save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        super().__init__(method, 
                         optimizer, 
                         num_epoch, 
                         dataloader, 
                         sheduler, 
                         update_teacher_after_n_epoch, 
                         update_teacher_each_n_step, 
                         save_model, 
                         save_model_each_n_epochs, 
                         project_root_or_url, 
                         project_name, 
                         run_name, 
                         logger, 
                         accumulate_grad, 
                         accumulate_step,
                         create_process_logger=False
                         )
        
        if self.is_main_rank:
            self.logger.info(f"Use FSDP. Number GPUs: {torch.cuda.device_count()}")
            self.process_logger = get_process_logger(logger, self.project_root, self.project_name, self.run_name)
            self.logger.info(f"{type(self.process_logger)} was created")


    def start_train_hook(self):
        if self.is_main_rank:
            super().start_train_hook()
    
    def start_epoch_hook(self, epoch):
        self.dataloader.sampler.set_epoch(epoch)

    def train_step(self, batch, do_step) -> dict:
        context = self.method.student.no_sync() if not do_step else nullcontext()
        
        # sync grad
        with context:
            loss_dict = self.method(batch)
            loss = loss_dict["loss"]
            loss = loss / self.accumulate_step
            loss.backward()

        if do_step or not self.accumulate_grad:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return loss_dict


    def update_teacher_weights(self):
        with FSDP.summon_full_params(self.method.student, writeback=False):
            with FSDP.summon_full_params(self.method.teacher, writeback=True):
                self.method.update_teacher_weights()


    def save_model_hook(self, model_save_step: int):
        if self.is_main_rank:
            student = self.method.student
            teacher = self.method.teacher
            
            with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, self.save_policy):
                student_state = student.state_dict()

            with FSDP.state_dict_type(teacher, StateDictType.FULL_STATE_DICT, self.save_policy):
                teacher_state = teacher.state_dict()

                self.model_copy.load_state_dict(student_state, strict=False)
                self.process_logger.save_model_state_dict(self.model_copy, "student_model", model_save_step)
                self.model_copy.load_state_dict(teacher_state, strict=False)
                self.process_logger.save_model_state_dict(self.model_copy, "teacher_model", model_save_step)
        dist.barrier()


    def end_train_hook(self):
        if self.is_main_rank:
            self.process_logger.end_experiment()


    def _update_step_history(self, d: dict):
        if self.is_main_rank:
            super()._update_step_history(d)


    def _do_log(self,
                epoch: int):
        if self.is_main_rank:
            super()._do_log(epoch)

