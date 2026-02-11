# Train modules

## Loggers

```python
class SSLProject.trainers.loggers.SimpleLogger(project_name: str, run_name: str, root: str|Path)
```
Class for logging train process locally

- project_name: name of project
- run_name: name of current run. If run with passed name alredy exists when will create new folder with name run_name_{index}
- root: root folder to save log information

#### Methods

```python
loglog(log: dict[str, float|int], step: int)
```
Loging train metrics


```python
write_log_in_csv(log: dict[str, float|int])
```
Write log info to csv table

```python
check_exists_file_or_create_new(path: Path|str, file_name: Path|str)
```
Create new folder if folder with name <code>file_name</code> alredy exists

```python
draw_plot()
```
Draw metric plot by data in csv table with metrics

```python
end_experiment()
```
plug


```python
class: SSLProject.trainers.loggers.SimpleMLFlowLogger(project_name: str, run_name: str, url: str)
```
Wrapper over the mlflow

#### Methods

```python
loglog(log: dict[str, float|int], step: int)
```
Loging train metrics

```python
end_experiment()
```
End mlflow experiment

---

## Trainers

```python
class SSLProject.trainers.simple_trainer.SimpleTrainer(method: BaseMethod, optimizer: torch.optim.Optimizer, num_epoch: int,
dataloader: Iterable, sheduler: torch.optim.lr_scheduler.LRScheduler, update_teacher_after_n_epoch: int=0,
update_teacher_each_n_step: int=1, project_root_or_url: str="", projrct_name: str="runs", run_name: str="ssl_run", logger: str="simple")
```

Class for train instances of <code>SSLProject.methods.base.BaseMethod</code>

- method: instance of <code>SSLProject.methods.base.BaseMethod</code>
- optimizer: Pytorch optimizer
- num_epoch: number epoche
- dataloader: Any iterable object. Must support call <code>__len__</code> method
- sheduler: Pytorch learning rate sheduler
- update_teacher_after_n_epoch: set number of epoch after which start update teacher weights. By default: 0
- update_teacher_each_n_step: sets the frequency of updating the teacher weights, if set 1, update each epoch. By default: 1
- project_root_or_url: root folder to save log information or url of mlflow. By default: ""
- projrct_name: name of project. By default: "runs"
- run_name: name of current run. By default: "ssl_run"
- logger: module for logging train process. Can use <code>simple</code> or <code>mlflow</code>. By default: <code>simple</code>

#### Methods

```python
train()
```

Run train process