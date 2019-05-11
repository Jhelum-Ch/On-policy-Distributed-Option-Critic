from pathlib import Path
import os

class DirectoryManager(object):
    # Level 1
    root = Path('./storage')

    def __init__(self, storage_name, seed, experiment_num=None):

        # Level 2
        self.storage_dir = DirectoryManager.root / storage_name

        if experiment_num is not None:
            self.current_experiment = f'experiment{experiment_num}'
        else:
            # Checks how many experiments already exist for this environment and storage_dir
            if not self.storage_dir.exists():
                self.current_experiment = 'experiment1'
            else:
                exst_run_nums = [int(str(folder.name).split('experiment')[1]) for folder in
                                 self.storage_dir.iterdir() if
                                 str(folder.name).startswith('experiment')]
                if len(exst_run_nums) == 0:
                    self.current_experiment = 'experiment1'
                else:
                    self.current_experiment = f'experiment{(max(exst_run_nums) + 1)}'

        # Level 3
        self.experiment_dir = self.storage_dir / self.current_experiment

        # Level 4
        self.seed_dir = self.experiment_dir / f"seed{seed}"

    def create_directories(self):
        os.makedirs(str(self.seed_dir), exist_ok=True)

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


def get_some_seeds(storage_dir, file_check='UNHATCHED'):
    # Finds all seed directories containing an UNHATCHED file and sorts them numerically
    sorted_experiments = DirectoryManager.get_all_experiments(storage_dir)

    some_seed_dirs = []
    for experiment_dir in sorted_experiments:
        some_seed_dirs += [seed_path for seed_path
                           in DirectoryManager.get_all_seeds(experiment_dir)
                           if (seed_path / file_check).exists()]

    return some_seed_dirs


def get_all_seeds(storage_dir):
    # Finds all seed directories and sorts them numerically
    sorted_experiments = DirectoryManager.get_all_experiments(storage_dir)

    all_seeds_dirs = []
    for experiment_dir in sorted_experiments:
        all_seeds_dirs += [seed_path for seed_path
                           in DirectoryManager.get_all_seeds(experiment_dir)]

    return all_seeds_dirs
