import random
import numpy
import torch
import collections
from math import floor, log10

def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d

def round_to_two(x):
    return round(x, -int(floor(log10(abs(x))) - 1))