# Here, for each hyperparam, enter the values you want to try in a list.
# All possible combinations of will be run as a separate experiment
import logging
VARIATIONS = [
    ('seed', [1, 123]),
    # Algorithmic params
    ('algo', ['a2c', 'ppo', 'oc']),
    #('lr', [7e-4]),
    #('entropy_coef', [0., 0.01]),
    ('value_loss_coef', [2.]),
    #('max_grad_norm', [0.5])
    #('gae_lambda', [0.95]),
    #('num_options', [1]),
    #('termination_loss_coef', [0.5]),
    #('termination_reg', [0.01]),
    #('tau', [0.01]),
    # Management params
    #('procs', [16]),
    ('frames', [5000]),
    #('log_interval', [1]),
    ('save_interval', [10]),
    # World params and Env params
    #('env', ['MiniGrid-DoorKey-5x5-v0']),
]

# Simple security check to make sure every specified parameter is defined only once
keys = [tuple[0] for tuple in VARIATIONS]
counted_keys = {key:keys.count(key) for key in keys}
for key in counted_keys.keys():
    if counted_keys[key] > 1:
        raise ValueError(f'Parameter "{key}" appears {counted_keys[key]} times in the schedule.')
