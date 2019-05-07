import os
from scripts.train import train
from utils.config import load_config_from_json
from utils.directory_tree import DirectoryManager
from utils.save import create_logger
from scripts.make_comparative_plots import create_comparative_figure
from pathlib import Path
import traceback
import datetime
import argparse
from multiprocessing import Process
import time
from utils import parse_bool
from tqdm import tqdm
import random

def get_run_schedule_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_dir', type=str, required=True)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--pbar', type=parse_bool, default=True)
    return parser.parse_args()

def run_schedule(serial_args, process_i=0):
    # Finds all directories containing an UNHATCHED file and sorts them numerically
    storage_dir = Path('storage') / serial_args.storage_dir

    def get_unhatched_seeds(storage_dir):
        sorted_experiments = DirectoryManager.get_all_experiments(storage_dir)

        all_unhatched_seeds = []
        for experiment_dir in sorted_experiments:
            all_unhatched_seeds += [seed_path for seed_path
                                    in DirectoryManager.get_all_seeds(experiment_dir)
                                    if (seed_path / 'UNHATCHED').exists()]

        return all_unhatched_seeds

    process_logger = create_logger(f"process_{process_i}", save_dir=storage_dir)

    # For each unhatched experiments, load the args and try to train the model
    call_i = 0
    while True:
        call_i += 1
        try:
            # random sleep time to delay different processes from one another
            time.sleep(random.uniform(0, 5))
            
            all_unhatched_seeds = get_unhatched_seeds(storage_dir)
            if len(all_unhatched_seeds) == 0:
                process_logger.info("No more experiments to run. Terminating.")
                break

            else:
                # gets the next experiment path
                seed_dir = all_unhatched_seeds[0]
                os.remove(str(seed_dir / 'UNHATCHED'))

                args = load_config_from_json(str(seed_dir / 'args.json'))
                dir_manager = DirectoryManager.init_from_seed_path(seed_dir)
                logger = create_logger(save_dir=dir_manager.seed_dir, streamHandle=False)

                if serial_args.pbar:
                    pbar = tqdm(position=process_i + (1 + serial_args.n_processes) * call_i)
                else:
                    pbar = None
                    process_logger.info(f'Starting {dir_manager.storage_dir.name}/{dir_manager.experiment_dir.name}/{dir_manager.seed_dir.name}')

                train(args, dir_manager, logger, pbar)

                if not serial_args.pbar:
                    process_logger.info(f'COMPLETED {dir_manager.storage_dir.name}/{dir_manager.experiment_dir.name}/{dir_manager.seed_dir.name}\n')
                open(str(seed_dir / 'COMPLETED'), 'w+').close()

        except Exception as e:
            if not serial_args.pbar:
                process_logger.info(f'CRASHED in {seed_dir.parent.parent.name}/{seed_dir.parent.name}/{seed_dir.name}\n')
            with open(str(seed_dir / 'CRASH.txt'), 'w+') as f:
                f.write(f'Crashed at : {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
                f.write(str(e))
                f.write(traceback.format_exc())

        try:
            # Creates a comparative figure showing total reward
            # for each episode of each agent of each experiment
            create_comparative_figure(storage_dir, create_logger(streamHandle=False))

        except Exception as e:
            if not serial_args.pbar:
                process_logger.info(f"WARNING: was not able to create comparative graphs")
            continue

if __name__ == '__main__':

    serial_args = get_run_schedule_args()
    print(f"\n\nRunning schedule for:"
          f"\n  storage_dir={serial_args.storage_dir}"
          f"\n  n_processes={serial_args.n_processes}\n\n")

    if serial_args.n_processes > 1:
        processes = []
        # create processes
        for i in range(serial_args.n_processes):
            processes.append(Process(target=run_schedule, args=(serial_args, i)))

        # start processes
        for p in processes:
            p.start()
            time.sleep(5)

        # wait for processes to end
        for p in processes:
            p.join()

        print("All processes are done. Closing '__main__'")

    else:
        run_schedule(serial_args)