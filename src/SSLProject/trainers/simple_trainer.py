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
from SSLProject.utils.utils import change_optimizer_and_sheduler
from SSLProject.trainers.loggers import SimpleLogger, SimpleMLFlowLogger
from SSLProject.methods.base import BaseMethod
from SSLProject.data_parallel import wrap_model_at_fsdp


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
                 logger: str="simple",
                 use_fsdp: bool=True, 
                 accumulate_grad: bool=True,
                 accumulate_step: int=1,
                 **kwargs):
        
        self.logger = get_logger("SimpleTrainer")

        # sharding the model if allowed
        self.use_fsdp = use_fsdp
        if use_fsdp:
            self.model_copy = copy.deepcopy(method.student).cpu()
            self.method = wrap_model_at_fsdp(method, **kwargs) 
            # self.method.to(f"cuda:0")
            self.save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            self.is_main_rank = self.use_fsdp and dist.get_rank() == 0
            if torch.cuda.is_available():
                self.logger.info(f"Use FSDP. Number GPUs: {torch.cuda.device_count()}")
            else:
                self.logger.info(f"Use FSDP. GPU is not allowed")
        else:
            self.method = method 
            self.logger.info(f"FSDP not use")
            self.use_fsdp = False
            self.is_main_rank = True

 
        # recreate optimizer and sheduler. Need if model was wrapped into FSDP module 
        self.optimizer, self.sheduler = change_optimizer_and_sheduler(self.method.student, optimizer, sheduler)
        self.dataloader = dataloader

        self.num_epoch = num_epoch
        self.accumulate_grad = accumulate_grad
        self.accumulate_step = accumulate_step if accumulate_grad else 1

        self.start_update = update_teacher_after_n_epoch
        self.update_each_n_step = update_teacher_each_n_step

        self.save_model = save_model
        self.save_model_each_n_epochs = save_model_each_n_epochs

        if self.is_main_rank:
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
        if self.is_main_rank:
            self.process_logger.start_experiment()
            self.logger.info("===== Start training =====") 
        step = 0
        batches_per_epoch = len(self.dataloader)

        try:
            
            for epoch in tqdm(range(self.num_epoch), desc="epoch: "):
                if self.use_fsdp:
                    self.dataloader.sampler.set_epoch(epoch)

                for batch in tqdm(self.dataloader, desc="batch progress: ", total=batches_per_epoch, leave=False):
                    step += 1
                    
                    do_step = (step % self.accumulate_step == 0)

                    context = self.method.student.no_sync() if self.use_fsdp and not do_step else nullcontext()

                    with context:
                        loss_dict = self.method(batch)
                        loss = loss_dict["loss"]
                        loss = loss / self.accumulate_step
                        loss.backward()

                    if do_step or not self.accumulate_grad:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    if epoch >= self.start_update and step % self.update_each_n_step == 0:
                        self.method.update_teacher_weights()
                    
                    if self.is_main_rank:
                        self._update_step_history(loss_dict)
                
                if self.accumulate_grad and (step % self.accumulate_step != 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.sheduler:
                    self.sheduler.step()

                if self.is_main_rank:
                    self._do_log(epoch)

                model_save_step = step//batches_per_epoch
                if self.save_model and (epoch+1) % self.save_model_each_n_epochs == 0 and self.is_main_rank:
                    student = self.method.student
                    teacher = self.method.teacher
                    if self.use_fsdp:
                        with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, self.save_policy):
                            student_state = student.state_dict()

                        with FSDP.state_dict_type(teacher, StateDictType.FULL_STATE_DICT, self.save_policy):
                            teacher_state = teacher.state_dict()

                        self.model_copy.load_state_dict(student_state, strict=False)
                        self.process_logger.save_model_state_dict(self.model_copy, "student_model", model_save_step)
                        self.model_copy.load_state_dict(teacher_state, strict=False)
                        self.process_logger.save_model_state_dict(self.model_copy, "teacher_model", model_save_step)
                    else:
                        self.process_logger.save_model_state_dict(student, "student_model", model_save_step)
                        self.process_logger.save_model_state_dict(teacher, "teacher_model", model_save_step)
                
                if dist.is_initialized():
                    dist.barrier()

        finally:
            if self.is_main_rank:
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


