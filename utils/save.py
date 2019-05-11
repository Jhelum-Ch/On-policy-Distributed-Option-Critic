import csv
import os
import torch
import json
import logging
import sys
import subprocess
import pickle

def get_model_path(save_dir):
    return os.path.join(save_dir, "model.pt")

def load_model(save_dir):
    path = get_model_path(save_dir)
    model = torch.load(path)
    model.eval()
    return model

def save_model(model, save_dir):
    path = get_model_path(save_dir)
    torch.save(model, path)

def get_graph_data_path(save_dir):
    return os.path.join(save_dir, "graph_data.pkl")

def save_graph_data(graph_data, save_dir):
    path = get_graph_data_path(save_dir)
    with open(path, 'wb') as f:
        pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_graph_data(save_dir):
    path = get_graph_data_path(save_dir)
    with open(path, 'rb') as f:
        graph_data = pickle.load(f)
    return graph_data

def get_status_path(save_dir):
    return os.path.join(save_dir, "status.json")

def load_status(save_dir):
    path = get_status_path(save_dir)
    with open(path) as file:
        return json.load(file)

def save_status(status, save_dir):
    path = get_status_path(save_dir)
    with open(path, "w") as file:
        json.dump(status, file)

def get_vocab_path(save_dir):
    return os.path.join(save_dir, "vocab.json")

def get_csv_path(save_dir):
    return os.path.join(save_dir, "log.csv")

def get_csv_writer(save_dir):
    csv_path = get_csv_path(save_dir)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def get_git_hash(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

def create_logger(name, loglevel, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(fmt=f'%(asctime)s - %(levelname)s - {name} - %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S', )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode='a'))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
