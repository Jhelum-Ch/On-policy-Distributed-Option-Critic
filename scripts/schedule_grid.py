# Here, for each hyperparam, enter the values you want to try in a list.
# All possible combinations of will be run as a separate experiment
import logging
VARIATIONS = [
    ('seed', [1, 22, 124]),
    # TEAMGRID
    # Algorithmic params
    ('algo', ['maddpg']),#['doc', 'a2c', 'ppo', 'oc', 'maddpg']),
    #('batch_size', [1, 16, 32, 64, 128, 256, 1024]),
    #('broadcast_penalty', [-0.01, -0.05, -0.1, 0.0]),
    ('lr', [7e-5, 7e-4, 7e-3, 1e-2]),
    ('entropy_coef', [0., 0.005, 0.01]),
    ('num_options', [1]),
    ('use_teamgrid',[True]),
    ('use_central_critic',[True]),
    ('use_always_broadcast',[True]),
    #('value_loss_coef', [2.]),
    ('max_grad_norm', [0.5,0.1,0.05]),
    #('gae_lambda', [0.95]),
    #('num_options', [1]),
    #('termination_loss_coef', [0.5]),
    #('termination_reg', [0.01]),
    #('tau', [0.01]),
    # Management params
    #('procs', [16]),
    ('frames', [3000000]),
    ('frames_per_proc', [30]),
    #('log_interval', [1]),
    #('save_interval', [10]),
    # World params and Env params
    ('env', ['TEAMGrid-FourRooms-v0']),
    #('scenario', ['simple_speaker_listener'])
]

# Simple security check to make sure every specified parameter is defined only once
keys = [tuple[0] for tuple in VARIATIONS]
counted_keys = {key:keys.count(key) for key in keys}
for key in counted_keys.keys():
    if counted_keys[key] > 1:
        raise ValueError(f'Parameter "{key}" appears {counted_keys[key]} times in the schedule.')
