import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import logging

try:
    import mlflow
except:
    pass 


class LoggerInterface:
    def loglog(self, log: dict[str, float|int], step:int):
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


    def end_experiment(self) -> None:
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

        self.experiment_is_run = False


    def loglog(self, 
               log: dict[str, float|int],
               step: int) -> None: 
        
        if not self.experiment_is_run:
            self.experiment_is_run = True
            mlflow.start_run(run_name=self.run_name)
            try:
                mlflow.system_metrics.log_system_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    sampling_interval=2,  # секунды
                    max_samples=100
                    )
            except:
                logging.info("System metrics not available")
            
        mlflow.log_metrics(log, step=step)


    def end_experiment(self) -> None:
        if mlflow.active_run():
            mlflow.end_run()
