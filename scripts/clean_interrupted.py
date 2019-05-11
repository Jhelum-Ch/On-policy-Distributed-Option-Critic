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
    all_seeds = get_all_seeds(storage_dir)
    unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')
    completed_seeds = get_some_seeds(storage_dir, file_check='COMPLETED')
    assert not any([seed_dir in completed_seeds for seed_dir in unhatched_seeds])

    if all([seed_dir in unhatched_seeds + completed_seeds for seed_dir in all_seeds]):
        print('Nothing to clean')
        return

    seeds_to_clean = [seed_dir for seed_dir in all_seeds if seed_dir not in unhatched_seeds + completed_seeds]
    for seed_dir in seeds_to_clean:
        print(f"Cleaning {seed_dir}")
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
    print(f'Done')


if __name__ == '__main__':
    serial_args = get_args()
    clean_interrupted(serial_args)
