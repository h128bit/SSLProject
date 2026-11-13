import csv
from pathlib import Path
import pandas as pd 

import torch
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics import Average
from ignite.handlers import ModelCheckpoint, global_step_from_engine

from SSLProject.methods.base import BaseMomentum


class DDPTrainer:
    def __init__(self, 
                 method: BaseMomentum,
                 optimizer,
                 scheduler,
                 update_teacher_after_n_epoch: int=0,
                 update_teacher_each_n_epoch: int=1,
                 run_path: str="run"
                 ):
        self.device = idist.device()

        self.update_teacher_after_n_epoch = update_teacher_after_n_epoch
        self.update_teacher_each_n_epoch = update_teacher_each_n_epoch

        self.method = method
        self.method.to(self.device)
        self.method = idist.auto_model(self.method)

        self.optimizer = idist.auto_optim(optimizer)
        self.scheduler = scheduler

        self.trainer_engine = Engine(self.train_step)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.step_scheduler)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.update_teacher_weights)
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, self.log_step)

        Average(output_transform=lambda x: x["loss"]).attach(self.trainer_engine, "avg_train_loss")

        if idist.get_rank() == 0:
            self.loss_keys = ["loss", "nnclr_loss", "centroid_loss", "red_loss"]

            self.log_path = Path(run_path)
            if self.log_path.exists():
                i = 1
                while True:
                    tmp = Path(str(self.log_path) + f"_{i}")
                    if tmp.exists():
                        i += 1
                    else:
                        break
                self.log_path = tmp 
            self.train_columns = ["epoch", "iteration"] + self.loss_keys
            self.log_table_path = self.log_path / "trsin_loss.csv"
            pd.DataFrame(columns=self.train_columns).to_csv(self.log_table_path, columns=self.train_columns, index=False)

            self._score_function = lambda engine: -engine.state.metrics["avg_train_loss"]
            model_checkpoint = ModelCheckpoint(
                self.log_path / "checkpoint",
                n_saved=1,
                filename_prefix="best",
                score_function=self._score_function,
                score_name="loss",
                global_step_transform=global_step_from_engine(self.trainer_engine), # helps fetch the trainer's state
                )
            self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {"model": self.method.module})


    def train_step(self, engine, batch):
        self.method.train()
        self.optimizer.zero_grad()

        batch[0] = batch[0].to(self.device).to(torch.float32)
        batch[1] = batch[1].to(self.device).to(torch.float32)

        with torch.autocast(device_type=str(self.device), dtype=torch.bfloat16):
            loss_dict = self.method(batch)
        loss = loss_dict["loss"]
        loss.backward()
        self.optimizer.step()

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k].item()

        return loss_dict


    def update_teacher_weights(self, engine):
        epoch = engine.state.epoch
        if epoch > self.update_teacher_after_n_epoch and epoch % self.update_teacher_each_n_epoch == 0:
            self.method.update_teacher_weights()


    def step_scheduler(self, engine):
        if self.scheduler:
            self.scheduler.step(engine.state.epoch - 1)


    def _write_in_csv_table(self, path, data):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)


    def log_step(self, engine):
        if idist.get_rank() == 0:
            epoch = engine.state.epoch
            iteration = engine.state.iteration
            output = engine.state.output
            loss_temp = ""
            loss_data = []
            for k in self.loss_keys:
                loss_temp += f"{k}: {output[k]:.4f} "
                loss_data.append(output[k])

            print(f"Epoch [{epoch}]: Iteration [{iteration}]: {loss_temp}")

            data = [epoch, iteration] + loss_data

            self._write_in_csv_table(self.log_table_path, data)
            