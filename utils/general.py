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
    d["mean"] = numpy.mean(array, axis=1)
    d["std"] = numpy.std(array, axis=1)
    d["min"] = numpy.amin(array, axis=1)
    d["max"] = numpy.amax(array, axis=1)
    return d

def round_to_two(x):
    return round(x, -int(floor(log10(abs(x))) - 1))
