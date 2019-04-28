from pathlib import Path
import os

class DirectoryManager(object):

    def __init__(self, storage_name, seed, experiment_num=None):
        # Level 1
        self.root = Path('./storage')

        # Level 2
        self.model_dir = self.root / storage_name

        if experiment_num is not None:
            self.current_experiment = f'experiment{experiment_num}'
        else:
            # Checks how many experiments already exist for this environment and storage_dir
            if not self.model_dir.exists():
                self.current_experiment = 'experiment1'
            else:
                exst_run_nums = [int(str(folder.name).split('experiment')[1]) for folder in
                                 self.model_dir.iterdir() if
                                 str(folder.name).startswith('experiment')]
                if len(exst_run_nums) == 0:
                    self.current_experiment = 'experiment1'
                else:
                    self.current_experiment = f'experiment{(max(exst_run_nums) + 1)}'

        # Level 3
        self.experiment_dir = self.model_dir / self.current_experiment

        # Level 4
        self.seed_dir = self.experiment_dir / f"seed{seed}"

        # Level 5
        self.recorders_dir = self.seed_dir / 'recorders'

    def create_directories(self):
        os.makedirs(str(self.recorders_dir), exist_ok=True)

    @staticmethod
    def get_all_experiments(storage_dir):
        all_experiments = [path for path in storage_dir.iterdir()
                           if path.is_dir() and str(path.stem).startswith('experiment')]

        return sorted(all_experiments, key=lambda item: (int(str(item.stem).strip('experiment')), item))

    @staticmethod
    def get_all_seeds(experiment_dir):
        all_seeds = [path for path in experiment_dir.iterdir()
                     if path.is_dir() and str(path.stem).startswith('seed')]

        return sorted(all_seeds, key=lambda item: (int(str(item.stem).strip('seed')), item))

    @classmethod
    def init_from_seed_path(cls, seed_path):
        assert isinstance(seed_path, Path)

        instance = cls(storage_name=seed_path.parents[1].name,
                       experiment_num=seed_path.parents[0].name.strip('experiment'),
                       seed=seed_path.name.strip('seed'))

        return instance
