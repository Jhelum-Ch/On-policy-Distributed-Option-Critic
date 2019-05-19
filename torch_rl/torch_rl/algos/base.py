from abc import ABC, abstractmethod
import torch
import numpy as np

from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 num_options=None, termination_loss_coef=None, termination_reg=None, option_epsilon=0.05):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        algo : str
            the name of the algorithm being used
            (will affect the way rollout and data collection are performed)
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        num_options : int
            the number of options
        termination_loss_coef : float
            the weight of the termination loss in the final objective
        termination_reg : float
            a small constant added to the option advantage to promote
            longer use of options (stretching)
        option_epsilon : float
            a small constant for the epsilon-soft policy over options
        """

        # Store parameters

        self.num_agents = num_agents
        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.num_actions = self.acmodel.num_actions
        self.num_options = num_options
        self.term_loss_coef = termination_loss_coef
        self.termination_reg = termination_reg
        self.option_epsilon = option_epsilon


        # Dimension convention

        self.batch_dim = 0
        self.opt_dim = 1
        self.act_dim = 2
        self.agt_dim = 3

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc + 1, self.num_procs)

        self.current_obss = self.env.reset()
        self.rollout_obss = [None]*(shape[0])

        if self.acmodel.recurrent:

            self.current_memories = [torch.zeros(shape[1], self.acmodel.memory_size, device=self.device) for _ in range(self.num_agents)]
            self.rollout_memories = [torch.zeros(*shape, self.acmodel.memory_size, device=self.device) for _ in range(self.num_agents)]

            if self.acmodel.use_central_critic:

                self.current_coord_memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
                self.rollout_coord_memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.current_mask = torch.ones(shape[1], device=self.device)
        self.rollout_masks = torch.zeros(*shape, device=self.device)
        self.rollout_actions = [torch.zeros(*shape, device=self.device, dtype=torch.int) for _ in range(self.num_agents)]
        self.rollout_rewards = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_advantages = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_log_probs = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

        self.rollout_options = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

        if self.acmodel.use_term_fn:

            self.rollout_terminates = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_terminates_prob = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

            self.current_options = [torch.randint(low=0, high=self.num_options, size=(shape[1],), device=self.device, dtype=torch.float) for _ in range(self.num_agents)]

        else:

            self.current_options = torch.arange(self.num_agents)

        if self.acmodel.use_act_values:

            self.rollout_values_swa = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_sw = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_s = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_sw_max = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

            self.rollout_targets = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

        else:

            self.rollout_values = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]


        if self.acmodel.use_broadcasting:

            self.current_broadcast_state = [torch.ones(shape[1], device=self.device) for _ in range(self.num_agents)]

            self.rollout_broadcast_masks = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_broadcast_probs = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

        # Initialize log values

        self.log_episode_return = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]
        self.log_episode_reshaped_return = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]
        self.log_episode_num_frames = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]

        self.log_done_counter = 0
        self.log_return = [[0] * shape[1] for _ in range(self.num_agents)]
        self.log_reshaped_return = [[0] * shape[1] for _ in range(self.num_agents)]
        self.log_num_frames = [[0] * shape[1] for _ in range(self.num_agents)]

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        if not self.recurrence:
            raise Exception("Deprecated: self.recurrence has to be True."
                            "If no reccurence is used, we will still have self.recurrence=True"
                            "but self.acmodel.use_memory will be set to False.")

        with torch.no_grad():

            rollout_length = len(self.rollout_obss)
            for i in range(rollout_length):

                agents_action = []
                agents_broadcast = []

                agents_act_dist = []
                agents_values = []
                agents_memory = []
                agents_term_dist = []
                agents_broadcast_dist = []
                agents_embedding = []

                for j, obs_j in enumerate(self.current_obss):

                    # Do one agent's forward propagation

                    preprocessed_obs = self.preprocess_obss(obs_j, device=self.device)

                    act_dist, values, memory, term_dist, broadcast_dist, embedding = self.acmodel(preprocessed_obs, self.current_memories[j] * self.current_mask.unsqueeze(1))

                    # collect outputs for each agent

                    agents_act_dist.append(act_dist)
                    agents_values.append(values)
                    agents_memory.append(memory)
                    agents_term_dist.append(term_dist)
                    agents_broadcast_dist.append(broadcast_dist)

                    # check broadcasting

                    if self.acmodel.use_broadcasting:

                        broadcast = broadcast_dist.sample()[range(self.num_procs), self.current_options[j].long()]

                        agents_broadcast.append(broadcast)
                        agents_embedding.append(broadcast.unsqueeze(1) * embedding)

                    # action selection

                    action = agents_act_dist[j].sample()[range(self.num_procs), self.current_options[j].long()]
                    agents_action.append(action)

                # compute option-action values with coordinator (doc)

                if self.acmodel.use_central_critic:

                    assert agents_values.count(None) == len(agents_values)
                    all_opt_act_values = torch.zeros((self.num_procs, self.num_options, self.num_actions, self.num_agents), device=self.device)
                    new_coord_memories = torch.zeros((self.num_procs, self.num_options, self.num_actions, self.acmodel.memory_size), device=self.device)

                    for j in range(self.num_agents):

                        for o in range(self.num_options):

                            for a in range(self.num_actions):
                                # TODO: big bottleneck here. these loops are extremely inefficient

                                option_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=o)
                                action_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=a)

                                option_idxs = [option_idxs_agent_j if k == j else self.current_options[k] for k in range(self.num_agents)]
                                action_idxs = [action_idxs_agent_j if k == j else agents_action[k] for k in range(self.num_agents)]
                                # TODO: because we need action here, action selection now happen before option selection. Make sure all the rest makes sense with that

                                values, coord_memory =  self.acmodel.forward_central_critic(agents_embedding,
                                                                                            option_idxs,
                                                                                            action_idxs,
                                                                                            self.current_coord_memory)
                                all_opt_act_values[:, o, a, j] = values
                                new_coord_memories[:, o, a, :] = coord_memory

                        agents_values[j] = all_opt_act_values[:, :, :, j]


                for j in range(self.num_agents):

                    # compute expectations over option-action values (which gives option-value Qsw and state-value Vs)

                    if self.acmodel.use_act_values:

                        Qsw_all             = torch.sum(agents_act_dist[j].probs * agents_values[j], dim=self.act_dim, keepdim=True)
                        Qsw_max, Qsw_argmax = torch.max(Qsw_all, dim=self.opt_dim, keepdim=True)
                        Qsw                 = Qsw_all[range(self.num_procs), self.current_options[j].long()]
                        Vs                  = Qsw_max

                    # option selection

                    if self.acmodel.use_term_fn:

                        assert self.acmodel.use_act_values

                        # check termination

                        terminate = agents_term_dist[j].sample()[range(self.num_procs), self.current_options[j].long()]

                        # changes option (for those that terminate)

                        random_mask = terminate * (torch.rand(self.num_procs) < self.option_epsilon).float()
                        chosen_mask = terminate * (1. - random_mask)
                        assert all(torch.ones(self.num_procs) == random_mask + chosen_mask + (1. - terminate))

                        random_options = random_mask * torch.randint(self.num_options, size=(self.num_procs,)).float()
                        chosen_options = chosen_mask * Qsw_argmax.squeeze().float()
                        self.current_options[j] = random_options + chosen_options + (1. - terminate) * self.current_options[j]

                    # update experience values (pre-step)

                    self.rollout_actions[j][i] = agents_action[j]
                    self.rollout_options[j][i] = self.current_options[j]
                    self.rollout_log_probs[j][i] = agents_act_dist[j].logits[range(self.num_procs), self.current_options[j].long(), action]

                    if self.acmodel.recurrent:
                        self.rollout_memories[j][i] = self.current_memories[j]
                        self.current_memories[j] = agents_memory[j]

                    if not self.acmodel.use_central_critic:

                        if self.acmodel.use_act_values:
                            self.rollout_values_swa[j][i] = agents_values[j][range(self.num_procs), self.current_options[j].long(), action].squeeze()
                            self.rollout_values_sw[j][i] = Qsw.squeeze()
                            self.rollout_values_s[j][i] = Vs.squeeze()
                            self.rollout_values_sw_max[j][i] = Qsw_max.squeeze()
                        else:
                            self.rollout_values[j][i] = agents_values[j][range(self.num_procs), self.current_options[j].long()].squeeze()

                    if self.acmodel.use_term_fn:

                        self.rollout_terminates_prob[j][i] = agents_term_dist[j].probs[range(self.num_procs), self.current_options[j].long()]
                        self.rollout_terminates[j][i] = terminate

                    if self.acmodel.use_broadcasting:

                        self.rollout_broadcast_probs[j][i] = agents_broadcast_dist[j].probs[range(self.num_procs), self.current_options[j].long()]
                        self.rollout_broadcast_masks[j][i] = agents_broadcast[j]


                if self.acmodel.recurrent and self.acmodel.use_central_critic:

                    self.rollout_coord_memories[i] = self.current_coord_memory
                    x = new_coord_memories[range(self.num_procs), self.current_options[j].long(), agents_action[j].long(), :]
                    self.current_coord_memory = x

                # environment step

                next_obss, rewards, done, _ = self.env.step(list(map(list, zip(*agents_action))))  # this list(map(list)) thing is used to transpose a list of lists

                # update experience values (post-step)

                self.rollout_obss[i] = self.current_obss
                self.current_obss = next_obss

                self.rollout_masks[i] = self.current_mask
                self.current_mask = 1. - torch.tensor(done, device=self.device, dtype=torch.float)

                self.log_done_counter = [0 for _ in range(self.num_agents)]

                for j, reward in enumerate(rewards):

                    if self.reshape_reward is not None:
                        # UNSUPPORTED for now
                        raise NotImplemented # TODO: figure out what is reshape_reward and support it
                        # self.rewards[j][i] = torch.tensor([
                        #     self.reshape_reward(obs_, action_, reward_, done_)
                        #     for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                        # ], device=self.device)
                    else:
                        self.rollout_rewards[j][i] = torch.tensor(reward, device=self.device)

                    if self.acmodel.use_term_fn:
                        # change current_option w.r.t. episode ending
                        self.current_options[j] = self.current_mask * self.current_options[j] + \
                                                  (1. - self.current_mask) * torch.randint(low=0, high=self.num_options,
                                                                                         size=(self.num_procs,),
                                                                                         device=self.device,
                                                                                         dtype=torch.float)

                    # update log values

                    self.log_episode_return[j] += torch.tensor(reward, device=self.device, dtype=torch.float)
                    self.log_episode_reshaped_return[j] += self.rollout_rewards[j][i]
                    self.log_episode_num_frames[j] += torch.ones(self.num_procs, device=self.device)

                    for k, done_ in enumerate(done):
                        if done_:
                            self.log_done_counter[j] += 1
                            self.log_return[j].append(self.log_episode_return[j][k].item())
                            self.log_reshaped_return[j].append(self.log_episode_reshaped_return[j][k].item())
                            self.log_num_frames[j].append(self.log_episode_num_frames[j][k].item())

                    self.log_episode_return[j] *= self.current_mask
                    self.log_episode_reshaped_return[j] *= self.current_mask
                    self.log_episode_num_frames[j] *= self.current_mask

            # Add advantage and return to experiences

            for j in range(self.num_agents):

                for i in reversed(range(self.num_frames_per_proc)):
                    if self.acmodel.use_term_fn and self.acmodel.use_act_values:

                        # target for q-learning objective

                        self.rollout_targets[j][i] = self.rollout_rewards[j][i] + \
                                         self.rollout_masks[i+1] * self.discount * \
                                         (
                                                 (1. - self.rollout_terminates_prob[j][i+1]) * self.rollout_values_sw[j][i+1] + \
                                                 self.rollout_terminates_prob[j][i+1] * self.rollout_values_sw_max[j][i+1]
                                         )

                        # option-advantage

                        self.rollout_advantages[j][i] = self.rollout_values_sw[j][i+1] - self.rollout_values_s[j][i+1]

                    elif not self.acmodel.use_term_fn and not self.acmodel.use_act_values:
                        next_mask = self.rollout_masks[i+1]
                        next_value = self.rollout_values[j][i+1]
                        next_advantage = self.rollout_advantages[j][i+1] if i < self.num_frames_per_proc else 0

                        delta = self.rollout_rewards[j][i] + self.discount * next_value * next_mask - self.rollout_values[j][i]
                        self.rollout_advantages[j][i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

                    else:
                        raise NotImplemented

            # Define experiences:
            #   the whole experience is the concatenation of the experience
            #   of each process.
            # In comments below:
            #   - T is self.num_frames_per_proc,
            #   - P is self.num_procs,
            #   - D is the dimensionality.

            exps = [DictList() for _ in range(self.num_agents)]
            logs = {k:[] for k in ["return_per_episode",
                                   "reshaped_return_per_episode",
                                   "num_frames_per_episode",
                                   "num_frames",
                                   "entropy",
                                   "value",
                                   "policy_loss",
                                   "value_loss",
                                   "grad_norm"]}

            for j in range(self.num_agents):
                exps[j].obs = [self.rollout_obss[i][j][k]
                            for k in range(self.num_procs)
                            for i in range(self.num_frames_per_proc)]
                if self.acmodel.recurrent:
                    # T x P x D -> P x T x D -> (P * T) x D
                    exps[j].memory = self.rollout_memories[j][:-1].transpose(0, 1).reshape(-1, *self.rollout_memories[j].shape[2:])
                    # T x P -> P x T -> (P * T) x 1
                    exps[j].mask = self.rollout_masks[:-1].transpose(0, 1).reshape(-1).unsqueeze(1)
                # for all tensors below, T x P -> P x T -> P * T
                exps[j].action = self.rollout_actions[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].reward = self.rollout_rewards[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].advantage = self.rollout_advantages[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].log_prob = self.rollout_log_probs[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].current_options = self.rollout_options[j][:-1].transpose(0, 1).reshape(-1).long()

                if self.acmodel.use_term_fn:

                    exps[j].terminate = self.rollout_terminates[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].terminate_prob = self.rollout_terminates_prob[j][:-1].transpose(0, 1).reshape(-1)

                    exps[j].target = self.rollout_targets[j][:-1].transpose(0, 1).reshape(-1)


                if self.acmodel.use_act_values:
                    exps[j].value_swa = self.rollout_values_swa[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].value_sw = self.rollout_values_sw[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].value_s = self.rollout_values_s[j][:-1].transpose(0, 1).reshape(-1)

                else:
                    exps[j].value = self.rollout_values[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].returnn = exps[j].value + exps[j].advantage

                # Preprocess experiences

                exps[j].obs = self.preprocess_obss(exps[j].obs, device=self.device)

                # Log some values

                keep = max(self.log_done_counter[j], self.num_procs)

                logs["return_per_episode"].append(self.log_return[j][-keep:])
                logs["reshaped_return_per_episode"].append(self.log_reshaped_return[j][-keep:])
                logs["num_frames_per_episode"].append(self.log_num_frames[j][-keep:])
                logs["num_frames"].append(self.num_frames)

                self.log_return[j] = self.log_return[j][-self.num_procs:]
                self.log_reshaped_return[j] = self.log_reshaped_return[j][-self.num_procs:]
                self.log_num_frames[j] = self.log_num_frames[j][-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
