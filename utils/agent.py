import torch
import numpy as np

import utils

class Agent:
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env_id, obs_space, save_dir, num_agents, option_epsilon, argmax=False, num_envs=1):
        _, self.preprocess_obss = utils.get_obss_preprocessor(env_id, obs_space, save_dir)
        self.acmodel = utils.load_model(save_dir)
        self.num_agents = num_agents
        assert self.acmodel.recurrent  # just to avoid supporting both
        self.argmax = argmax
        self.num_envs = num_envs
        self.num_options = self.acmodel.num_options
        self.option_epsilon = option_epsilon
        #self.current_options = current_options
        #self.current_options = torch.arange(self.num_agents)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.acmodel.recurrent:
            #print(self.acmodel.memory_size)
            # if isinstance(self.acmodel.memory_size, int):
            #     self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size)
            #     self.memories = self.memories.unsqueeze(0)
            # elif isinstance(self.acmodel.memory_size, list):
            #     self.memories = [torch.zeros(self.num_envs, self.acmodel.memory_size[j]) for j in range(self.num_agents)]
            #     self.memories = self.memories.unsqueeze(0)

            self.memories = [torch.zeros(self.num_envs, self.acmodel.memory_size) for _ in range(self.num_agents)]
            #self.memories = self.memories.unsqueeze(0)
    def get_actions(self, obss):

        options = []
        actions = []
        broadcasts = []

        for j, obs in enumerate(obss):

            # proprocess observations

            preprocessed_obss = self.preprocess_obss(obs)

            # forward propagation

            with torch.no_grad():
                #act_dist, values, memory, term_dist, _ = self.acmodel(preprocessed_obss, self.memories)
                #if self.acmodel.use_teamgrid:
                if self.num_options is not None:
                    if not self.acmodel.always_broadcast:

                        act_mlp, act_dist, values, values_b, memory, term_dist, broadcast_dist, embedding = \
                            self.acmodel.forward_agent_critic(preprocessed_obss, self.memories[j],j)

                        #agents_values_b.append(values_b)
                        #agents_broadcast_dist.append(broadcast_dist)

                    else:

                        act_mlp, act_dist, values, memory, term_dist, embedding = \
                            self.acmodel.forward_agent_critic(preprocessed_obss, self.memories[j],j)
                else:
                    if not self.acmodel.always_broadcast:
                        act_mlp, act_dist, values, values_b, memory, broadcast_dist, embedding = \
                            self.acmodel.forward_agent_critic(preprocessed_obss, self.memories[j],j)
                        #agents_values_b.append(values_b)
                        #agents_broadcast_dist.append(broadcast_dist)
                    else:
                        act_mlp, act_dist, values, memory, embedding = \
                            self.acmodel.forward_agent_critic(preprocessed_obss, self.memories[j],j)



            if self.acmodel.use_term_fn:

                # option selection

                self.current_options = [
                    torch.randint(low=0, high=self.num_options, size=(1,), device=self.device,
                                  dtype=torch.float) for _ in range(self.num_agents)]


                terminate = term_dist.sample()[:, self.current_options[j].long()] #use self.current_options[j] if using DOC/OC

                # if terminate:
                #     self.current_options = int(torch.argmax(torch.sum(act_dist.probs * values, dim=-1), dim=-1))
                # changes option (for those that terminate)

                random_mask = terminate * (torch.rand(1) < self.option_epsilon).float()
                chosen_mask = terminate * (1. - random_mask)
                #assert all(torch.ones(self.num_procs) == random_mask + chosen_mask + (1. - terminate))

                random_options = random_mask * torch.randint(self.num_options, size=(1,)).float()
                self.current_options[j] = random_options
                #print('self.current_options', self.current_options)
                # chosen_options = chosen_mask * Qsw_coord_argmax.squeeze().float()
                # chosen_options = chosen_mask * Qsw_argmax.squeeze().float()
                # self.current_options[j] = random_options + chosen_options + (1. - terminate) * self.current_options[j]

            # action selection

            if self.argmax:
                action = act_dist.probs.argmax(dim=-1, keepdim=True)
                broadcast = broadcast_dist.probs.argmax(dim=-1, keepdim=True)
            else:
                action = act_dist.sample()[:,self.current_options[j].long()]
                broadcast = broadcast_dist.sample()[:,self.current_options[j].long(), action].squeeze()

            #action = action[:, self.current_options].squeeze() #use self.current_options[j] if using DOC/OC

            if torch.cuda.is_available():
                action = action.cpu().numpy()
                broadcast = broadcast.cpu().numpy()

            #print('self.current_options', self.current_options)
            options.append(self.current_options)
            actions.append(action)
            broadcasts.append(broadcast)

        return options, actions, broadcasts

    def get_action(self, obss):
        new_obss = []
        for obs in obss:
            new_obss.append([obs])
        return self.get_actions(new_obss)

    def analyze_feedbacks(self, rewards, done):
        if not self.acmodel.use_teamgrid:
            done = all(done)
            #print('done', done)
        if self.acmodel.recurrent:
            masks = 1. - torch.tensor(done, dtype=torch.float) #.unsqueeze(1)
            #print('masks', masks)
            for j in range(self.num_agents):
                self.memories[j] *= masks
            #print('memories', self.memories)

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])