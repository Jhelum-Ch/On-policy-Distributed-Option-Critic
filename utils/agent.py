import torch

import utils

class Agent:
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env_id, obs_space, model_dir, argmax=False, num_envs=1, num_options=1):
        _, self.preprocess_obss = utils.get_obss_preprocessor(env_id, obs_space, model_dir)
        self.acmodel = utils.load_model(model_dir)
        self.argmax = argmax
        self.num_envs = num_envs
        self.num_options = num_options
        self.curr_option = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size)

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss)

        with torch.no_grad():
            if self.acmodel.recurrent and self.num_options > 1:
                action, act_dist, _, memory, term_dist = self.acmodel(preprocessed_obss, self.memories)
            elif self.acmodel.recurrent and not self.num_options > 1:
                action, act_dist, _, memory = self.acmodel(preprocessed_obss, self.memories)
            elif not self.acmodel.recurrent and self.num_options > 1:
                action, act_dist, _, term_dist = self.acmodel(preprocessed_obss)
            else:
                action, act_dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = act_dist.probs.max(1, keepdim=True)[1]
        else:
            actions = action

        if torch.cuda.is_available():
            actions = actions.cpu().numpy()

        return actions

    def get_action(self, obs):
        return self.get_actions([obs]).item()

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])