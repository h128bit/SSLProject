import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch

try:
    import mlflow
    from mlflow.system_metrics import enable_system_metrics_logging
    enable_system_metrics_logging()
except:
    pass 


class LoggerInterface:
    def loglog(self, log: dict[str, float|int], step:int):
        raise NotImplementedError
    
    def save_model_state_dict(self, model: torch.nn.Module, model_name: str, step:int):
        raise NotImplementedError
    
    def start_experiment(self):
        raise NotImplementedError

    def end_experiment(self):
        raise NotImplementedError



class SimpleLogger(LoggerInterface):
    def __init__(self,
                 project_name: str,
                 run_name: str,
                 root: str|Path):
        super().__init__()

        self.root = Path(root)
        self.prj_name = project_name
        self.run_name = run_name

        self.path_to_prj = self.root/self.prj_name

        if not self.path_to_prj.exists():
            self.path_to_prj.mkdir()

        self.path_to_run = self.check_exists_file_or_create_new(self.path_to_prj, self.run_name)
        self.path_to_run.mkdir()

        self.plot_folder_path = self.path_to_run/"plots"
        if not self.plot_folder_path.exists():
            self.plot_folder_path.mkdir()

        self.path_to_save_model = self.path_to_run / "artifacts"
        if not self.path_to_save_model.exists():
            self.path_to_save_model.mkdir()

        self.log_file_path = self.check_exists_file_or_create_new(self.path_to_run, "losses.csv")
        self.first_write = True


    def write_log_in_csv(self, 
                         log: dict[str, float|int]) -> None:
        pd.DataFrame(data=log, index=[0]).to_csv(path_or_buf=self.log_file_path, index=False, header=self.first_write, mode="a")
        self.first_write = False

    
    def check_exists_file_or_create_new(self, 
                                        path: Path|str, 
                                        file_name: Path|str) -> Path:

        path = Path(path)
        file_name = Path(file_name)
        ext = file_name.suffix
        name = file_name.stem

        idx = 0
        file_path = path / file_name
        while file_path.exists():
            file_name = name + f"_{idx}" + ext
            file_path = path/file_name
            idx += 1

        return file_path
        

    def draw_plot(self) -> None:
        plt.ioff()

        df = pd.read_csv(self.log_file_path)
        cols = df.columns

        fig, axs = plt.subplots(ncols=len(cols), nrows=1, figsize=(20, 5))
        axs = axs.flatten()

        for col, ax in zip(cols, axs):
            vals = df[col].values
            ax.plot(vals)
            ax.set_title(col)

        plt.tight_layout()
        fig.savefig(self.plot_folder_path/"plot.png", bbox_inches='tight', dpi=200)

        plt.close(fig)
            

    def loglog(self, 
               log: dict[str, float|int],
               step: int) -> None:
        self.write_log_in_csv(log)
        self.draw_plot()


    def save_model_state_dict(self, model: torch.nn.Module, model_name: str, step:int):
        path = self.path_to_save_model / f"{model_name}.pt"
        torch.save(model.state_dict(), path)


    def end_experiment(self) -> None:
        pass

    
    def start_experiment(self):
        pass


class SimpleMLFlowLogger(LoggerInterface):
    def __init__(self,
                 project_name: str,
                 run_name: str,
                 url: str):
        import mlflow # check what mlflow is install

        self.prj_name = project_name
        self.run_name = run_name

        mlflow.set_tracking_uri(url)
        mlflow.set_experiment(self.prj_name)


    def loglog(self, 
               log: dict[str, float|int],
               step: int) -> None: 
            
        mlflow.log_metrics(log, step=step)


    def save_model_state_dict(self, model: torch.nn.Module, model_name: str, step:int):
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=model_name,
            step=step
        )


    def start_experiment(self):
        runs = mlflow.search_runs(
            experiment_names=[self.prj_name],
            max_results=1000            
            )
        
        try:
            runs = runs["tags.mlflow.runName"].to_list()
        except KeyError:
            runs = []
         
        idx = 1
        tmp_name = self.run_name
        while tmp_name in runs:
            tmp_name = f"{self.run_name}_{idx}"
            idx += 1
        self.run_name = tmp_name
        mlflow.start_run(run_name=self.run_name)


    def end_experiment(self) -> None:
        if mlflow.active_run():
            mlflow.end_run()
