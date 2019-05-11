import logging
import json
import argparse

def parse_bool(bool_arg):
    """
    Parse a string representing a boolean.
    :param bool_arg: The string to be parsed
    :return: The corresponding boolean
    """
    if bool_arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif bool_arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean argument expected. Got {bool_arg} instead.')


def parse_log_level(level_arg):
    """
    Parse a string representing a log level.
    :param level_arg: The desired level as a string (ex: 'info').
    :return: The corresponding level as a `logging` member (ex: `logging.INFO`).
    """
    return getattr(logging, level_arg.upper())


def load_dict_from_json(filename):
    """
    Loads a python dictionary from a json file
    :param filename: full filename to json file from which to load the config
    :return: dictionary object with content from the json file
    """
    with open(filename, 'r') as f:
        loaded_dict = json.load(f)

    return loaded_dict


def save_dict_to_json(dictionary, filename):
    """
    Saves a python dictionary to json file
    :param dictionary: dictionary object
    :param config: full filename to json file in which to save config
    :return: None
    """
    with open(filename, 'w+') as f:
        json.dump(dictionary, f)


def save_config_to_json(config, filename):
    """
    Saves a set of commandline arguments (config) to json file
    :param config: parser.parse_args() object
    :param config: full filename to json file in which to save config
    :return: None
    """
    with open(filename, 'w+') as f:
        json.dump(vars(config), f)


def load_config_from_json(filename):
    """
    Creates a Namaspace object and populates it using dictionary from a json file
    :param filename: full filename to json file from which to load the config
    :return: argparse.ArgumentParser.parse_args() object (NameSpace) populated as in the json file
    """
    with open(filename, 'r') as f:
        loaded_config_dict = json.load(f)

    # Creates a pointer to default NameSpace dict
    config = argparse.ArgumentParser().parse_args(args="")
    config_dict = vars(config)

    # Updates the default dict using the one loaded from the json file
    config_dict.update(loaded_config_dict)

    return config


def config_to_str(config):
    config_string = 'Configs'
    for arg in vars(config):
        config_string += f'\n    {arg}: {getattr(config, arg)}'
    return config_string
