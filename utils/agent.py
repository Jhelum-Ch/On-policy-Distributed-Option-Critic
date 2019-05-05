import torch
import numpy as np

import utils

class Agent:
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env_id, obs_space, save_dir, num_agents, argmax=False, num_envs=1):
        _, self.preprocess_obss = utils.get_obss_preprocessor(env_id, obs_space, save_dir)
        self.acmodel = utils.load_model(save_dir)
        self.num_agents = num_agents
        assert self.acmodel.recurrent  # just to avoid supporting both
        self.argmax = argmax
        self.num_envs = num_envs
        self.num_options = self.acmodel.num_options
        self.current_options = torch.arange(self.num_agents)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size)

    def get_actions(self, obss):

        actions = []

        for j, obs in enumerate(obss):

            # proprocess observations

            preprocessed_obss = self.preprocess_obss(obs)

            # forward propagation

            with torch.no_grad():
                act_dist, values, memory, term_dist = self.acmodel(preprocessed_obss, self.memories)

            if self.acmodel.use_term_fn:

                # option selection

                terminate = term_dist.sample()[:, self.current_options[j]]
                if terminate:
                    self.current_options = int(torch.argmax(torch.sum(act_dist.probs * values, dim=-1), dim=-1))

            # action selection

            if self.argmax:
                action = act_dist.probs.argmax(dim=-1, keepdim=True)
            else:
                action = act_dist.sample()

            action = action[:, self.current_options[j]].squeeze()

            if torch.cuda.is_available():
                action = action.cpu().numpy()

            actions.append(action)

        return actions

    def get_action(self, obss):
        new_obss = []
        for obs in obss:
            new_obss.append([obs])
        return self.get_actions(new_obss)

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])