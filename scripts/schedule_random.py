
# Here, for each hyperparam, enter the function that you want the random-search to sample from.
# For each experiment, a set of hyperparameters will be sampled using these functions

# Examples:
# int:          np.random.randint(low=64, high=512)
# float:        np.random.uniform(low=-3., high=1.)
# bool:         bool(np.random.binomial(n=1, p=0.5))
# exp_float:    10.**np.random.uniform(low=-3., high=1.)
# fixed_value:  fixed_value

import numpy as np
from utils.general import round_to_two

N_SEEDS = 3
SEED_VARIATIONS = [i * 10 for i in range(N_SEEDS)]

def sample_experiment():
    sampled_config = [
        ('seed', 1),
        # Algorithmic params
        ('algo', 'oc'),
        ('lr', round_to_two(10**np.random.uniform(low=-6., high=-2))),
        #('entropy_coef', 0.01),
        ('value_loss_coef', np.random.uniform(low=0.1, high=10.)),
        #('max_grad_norm', 0.5),
        #('gae_lambda', 0.95),
        #('num_options', np.random.randint(low=1, high=9)),
        #('termination_loss_coef', 0.5),
        #('termination_reg', 0.01),
        #('tau', 0.01),
        # Management params
        #('procs', 16),
        ('frames', 10000),
        #('log_interval', 1),
        #('save_interval', 10),
        # World params and Env params
        #('env', 'MiniGrid-DoorKey-5x5-v0'),
    ]

    # Simple security check to make sure every specified parameter is defined only once
    keys = [tuple[0] for tuple in sampled_config]
    counted_keys = {key: keys.count(key) for key in keys}
    for key in counted_keys.keys():
        if counted_keys[key] > 1:
            raise ValueError(f'Parameter "{key}" appears {counted_keys[key]} times in the schedule.')

    return sampled_config
