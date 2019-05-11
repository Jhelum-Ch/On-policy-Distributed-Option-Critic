import sys

import argparse

sys.path.append('..')
from utils.directory_tree import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_dir', type=str, required=True)
    return parser.parse_args()


def clean_interrupted(args):
    storage_dir = DirectoryManager.root / args.storage_dir
    all_seeds = set(get_all_seeds(storage_dir))
    all_unhatched_seeds = set(get_some_seeds(storage_dir, file_check='UNHATCHED'))
    all_completed_seeds = set(get_some_seeds(storage_dir, file_check='COMPLETED'))
    assert not all_completed_seeds & all_unhatched_seeds

    if (all_unhatched_seeds | all_completed_seeds) >= all_seeds:
        print('Nothing to clean')
        return

    seeds_to_clean = all_seeds - (all_unhatched_seeds | all_completed_seeds)
    for seed_dir in seeds_to_clean:
        print(seed_dir)
        paths = seed_dir.iterdir()
        for path in paths:
            if path.is_dir() and path.name in ["recorders", "incremental"]:
                history_files = path.iterdir()
                for history_file in history_files:
                    os.remove(str(history_file))
            elif path.name not in ["config.json", "config_unique.json"]:
                os.remove(str(path))
            else:
                pass

        open(str(seed_dir / 'UNHATCHED'), 'w+').close()
    print(f'Done\n{args.storage_dir} cleaned')


if __name__ == '__main__':
    serial_args = get_args()
    clean_interrupted(serial_args)
