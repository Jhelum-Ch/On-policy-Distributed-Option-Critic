import sys
sys.path.append('..')
from scripts.train import train
from utils.config import load_config_from_json
from utils.directory_tree import *
from utils.save import create_logger
from scripts.make_comparative_plots import create_comparative_figure
import numpy as np
import traceback
import datetime
import argparse
from multiprocessing import Process
import time
import logging
from utils.config import parse_bool
from tqdm import tqdm
# from serial.benchmark import compare_models_on_bar_chart, compare_models_by_learning_curve

def get_run_schedule_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_dir', type=str, required=True)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--n_experiments_per_proc', type=int, default=np.inf)
    parser.add_argument('--use_pbar', type=parse_bool, default=False)

    return parser.parse_args()

def run_schedule(args, master_logger, process_i=0):
    try:

        storage_dir = DirectoryManager.root / args.storage_dir
        all_unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')

        # For each unhatched experiments, load the config and try to train the model
        call_i = 0
        while len(all_unhatched_seeds) > 0:

            call_i += 1
            if call_i > args.n_experiments_per_proc:
                break

            try:
                all_unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')
                seed_dir = all_unhatched_seeds[0]
                os.remove(str(seed_dir / 'UNHATCHED'))

            except (IndexError, FileNotFoundError):
                master_logger.info(f"PROCESS{process_i} - It seems like there are no more unhatched seeds.")
                break

            try:
                config = load_config_from_json(str(seed_dir / 'config.json'))
                dir_manager = DirectoryManager.init_from_seed_path(seed_dir)

                experiment_logger = create_logger(
                    name=f'PROCESS{process_i}:{dir_manager.storage_dir.name}/{dir_manager.experiment_dir.name}/{dir_manager.seed_dir.name}',
                    loglevel=logging.INFO,
                    logfile=dir_manager.seed_dir / 'logger.out',
                    streamHandle=not(args.use_pbar)
                    )

                if args.use_pbar:
                    pbar = tqdm(position=process_i + (1 + args.n_processes) * call_i)
                    pbar.desc = f"PROCESS{process_i}:"
                else:
                    pbar = None

                master_logger.info(f"PROCESS{process_i} - {dir_manager.storage_dir.name}/{dir_manager.experiment_dir.name}/{dir_manager.seed_dir.name} - Launching...")
                train(config, dir_manager, experiment_logger, pbar)

                open(str(seed_dir / 'COMPLETED'), 'w+').close()

            except Exception as e:
                with open(str(seed_dir / 'CRASH.txt'), 'w+') as f:
                    f.write(f'Crashed at : {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
                    f.write(str(e))
                    f.write(traceback.format_exc())

        all_seeds = get_all_seeds(storage_dir)
        unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')
        crashed_seeds = get_some_seeds(storage_dir, file_check='CRASH.txt')
        completed_seeds = get_some_seeds(storage_dir, file_check='COMPLETED')

        if len(unhatched_seeds) == 0 and (len(crashed_seeds) + len(completed_seeds) == len(all_seeds)):
            master_logger.info(f"PROCESS{process_i} - Finalizing...")

            try:
                create_comparative_figure(storage_dir, logger=master_logger)
            except Exception as e:
                master_logger.info(f"PROCESS{process_i} - \n{type(e)}: unable to plot comparative graphs\n\n{e}\n{traceback.format_exc()}")

            # # If all experiments are completed benchmark them
            # if all_seeds == completed_seeds:
            #
            #     try:
            #         compare_models_on_bar_chart(env_dir=args.env_id, model_names=[args.model_name], n_episodes=5, agent_i=0, normalize_with_first_model=False, num_key=True, sort_bars=True, logger=master_logger)
            #     except Exception as e:
            #         master_logger.info(f"PROCESS{process_i} - \n{type(e)}: unable to benchmark performance\n\n{e}\n{traceback.format_exc()}")
            #
            #     try: compare_models_by_learning_curve(env_dir=args.env_id, model_names=[args.model_name], agent_i=0, num_key=True, n_labels=10, logger=master_logger)
            #     except Exception as e:
            #         master_logger.info(f"PROCESS{process_i} - \n{type(e)}: unable to benchmark learning\n{e}\n{traceback.format_exc()}")

        master_logger.info(f"PROCESS{process_i} - Done. Shutting down.")

    except Exception as e:
        master_logger.info(f"PROCESS{process_i} - The process CRASHED with the following error:\n{e}")

    return


if __name__ == '__main__':

    serial_args = get_run_schedule_args()
    storage_dir = DirectoryManager.root / serial_args.storage_dir

    master_logger = create_logger(name=f'RUN_SCHEDULE:', loglevel=logging.INFO,
                                  logfile=storage_dir / 'run_schedule_logger.out', streamHandle=False)

    master_logger.info("="*200)
    master_logger.info("="*200)
    master_logger.info(f"\n\nRunning schedule for:"
                       f"\n  storage_dir={serial_args.storage_dir}"
                       f"\n  n_processes={serial_args.n_processes}"
                       f"\n  n_experiments_per_proc={serial_args.n_experiments_per_proc}"
                       f"\n  use_pbar={serial_args.use_pbar}\n\n")

    if serial_args.n_processes > 1:
        processes = []
        # create processes
        for i in range(serial_args.n_processes):
            processes.append(Process(target=run_schedule, args=(serial_args, master_logger, i)))

        try:
            # start processes
            for p in processes:
                p.start()
                time.sleep(3)

            # waits for all processes to end
            dead_processes = []
            while any([p.is_alive() for p in processes]):

                # check if some processes are dead
                for i, p in enumerate(processes):
                    if not p.is_alive() and i not in dead_processes:
                        master_logger.info(f'PROCESS{i} has died.')
                        dead_processes.append(i)

                time.sleep(3)

        except KeyboardInterrupt:
            master_logger.info("KEYBOARD INTERRUPT\nKilling all processes")
            # terminates all processes
            for process in processes:
                process.terminate()

        master_logger.info("All processes are done. Closing '__main__'\n\n")

    else:
        run_schedule(serial_args, master_logger)
