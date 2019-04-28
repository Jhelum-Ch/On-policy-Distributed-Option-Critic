import torch
import numpy as np

import utils

class Agent:
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env_id, obs_space, save_dir, argmax=False, num_envs=1):
        _, self.preprocess_obss = utils.get_obss_preprocessor(env_id, obs_space, save_dir)
        self.acmodel = utils.load_model(save_dir)
        assert self.acmodel.recurrent  # just to avoid supporting both
        self.argmax = argmax
        self.num_envs = num_envs
        self.num_options = self.acmodel.num_options
        self.curr_option = np.random.randint(self.num_options) if self.num_options is not None else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size)

    def get_actions(self, obss):

        # proprocess observations

        preprocessed_obss = self.preprocess_obss(obss)

        # forward propagation

        with torch.no_grad():
            if self.num_options is not None:
                act_dist, act_values, memory, term_dist = self.acmodel(preprocessed_obss, self.memories)
            else:
                act_dist, _, memory = self.acmodel(preprocessed_obss, self.memories)

        if self.num_options is not None:

            # option selection

            terminate = term_dist.sample()[:, self.curr_option]
            if terminate:
                self.curr_option = int(torch.argmax(torch.sum(act_dist.probs * act_values, dim=-1), dim=-1))

        # action selection

        if self.argmax:
            action = act_dist.probs.max(axis=-1, keepdim=True)[1]
        else:
            action = act_dist.sample()

        if self.num_options is not None:
            action = action[:, self.curr_option]

        if torch.cuda.is_available():
            action = action.cpu().numpy()

        return action

    def get_action(self, obs):
        return self.get_actions([obs]).item()

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])