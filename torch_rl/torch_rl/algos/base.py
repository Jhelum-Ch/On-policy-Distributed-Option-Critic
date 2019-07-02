from abc import ABC, abstractmethod
import torch
import numpy as np
import itertools
import copy

from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, broadcast_penalty,
                 num_options=None, termination_loss_coef=None, termination_reg=None, option_epsilon=0.05,
                 always_broadcast=False):
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
        #self.recurrence_coord = recurrence_coord
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.broadcast_penalty = broadcast_penalty
        self.num_actions = self.acmodel.num_actions
        #self.num_options = num_options
        self.num_options = self.acmodel.num_options
        self.term_loss_coef = termination_loss_coef
        self.termination_reg = termination_reg
        self.option_epsilon = option_epsilon
        self.always_broadcast = always_broadcast


        # Dimension convention

        self.batch_dim = 0
        self.opt_dim = 1
        self.act_dim = 2
        #self.brd_dim = 3
        self.agt_dim = 3

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Control parameters

        # assert self.acmodel.recurrent or self.recurrence_agents == 1 and recurrence_coord == 1
        # assert self.num_frames_per_proc % self.recurrence_agents == 0 and self.num_frames_per_proc % self.recurrence_coord == 0
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc + 1, self.num_procs)

        self.current_obss = self.env.reset()
        self.rollout_obss = [None]*(shape[0])


        if self.acmodel.recurrent:

            self.current_agent_memories = [torch.zeros(shape[1], self.acmodel.memory_size, device=self.device) for _ in range(self.num_agents)]
            self.rollout_agent_memories = [torch.zeros(*shape, self.acmodel.memory_size, device=self.device) for _ in range(self.num_agents)]


            if self.acmodel.use_central_critic:

                self.current_coord_memory = torch.zeros(shape[1], self.acmodel.coord_memory_size, device=self.device)
                self.rollout_coord_memories = torch.zeros(*shape, self.acmodel.coord_memory_size, device=self.device)
                self.rollout_coord_advantages = torch.zeros(*shape, device=self.device)

        self.current_mask = torch.ones(shape[1], device=self.device)
        self.rollout_masks = torch.zeros(*shape, device=self.device)
        self.rollout_masked_embeddings = [torch.zeros(*shape, self.acmodel.semi_memory_size, device=self.device) for _ in range(self.num_agents)]


        self.rollout_actions = [torch.zeros(*shape, device=self.device, dtype=torch.int) for _ in range(self.num_agents)]
        self.rollout_rewards = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_rewards_plus_broadcast_penalties = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_advantages = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_advantages_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_log_probs = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_broadcast_log_probs = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

        self.rollout_options = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

        if self.acmodel.use_term_fn:

            self.rollout_terminates = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_terminates_prob = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

        if self.num_options is not None:
            self.current_options = [torch.randint(low=0, high=self.num_options, size=(shape[1],), device=self.device, dtype=torch.float) for _ in range(self.num_agents)]
           # self.current_joint_option = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]

        else:

            self.current_options = torch.arange(self.num_agents)

        self.current_joint_action = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]

        if self.acmodel.use_act_values:
            if self.acmodel.use_central_critic:
                self.rollout_coord_value_swa = torch.zeros(*shape, device=self.device)
                self.rollout_coord_value_sw = torch.zeros(*shape, device=self.device)
                self.rollout_coord_value_s = torch.zeros(*shape, device=self.device)
                self.rollout_coord_value_sw_max = torch.zeros(*shape, device=self.device)

                self.rollout_coord_target = torch.zeros(*shape, device=self.device)


            self.rollout_values_swa = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_sw = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]


            self.rollout_values_s = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_sw_max = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

            self.rollout_targets = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_targets_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]


        else:

            self.rollout_values = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]


        if self.acmodel.use_broadcasting:

            self.current_broadcast_state = [torch.ones(shape[1], device=self.device) for _ in range(self.num_agents)]

            self.rollout_values_swa_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_sw_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_s_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_sw_b_max = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_advantages_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]

            self.rollout_broadcast_masks = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]
            #self.rollout_broadcast_probs = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]


        # Initialize log values

        self.log_episode_return = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]
        self.log_episode_return_with_broadcast_penalties = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]
        self.log_episode_reshaped_return = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]
        self.log_episode_num_frames = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]

        self.log_done_counter = 0
        self.log_return = [[0] * shape[1] for _ in range(self.num_agents)]
        self.log_return_with_broadcast_penalties = [[0] * shape[1] for _ in range(self.num_agents)]
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
                            "but self.acmodel.use_memory_agents will be set to False.")

        with torch.no_grad():

            rollout_length = len(self.rollout_obss)

            for i in range(rollout_length):

                #print('i', i)
                self.rollout_obss[i] = self.current_obss


                # FOR CENTRALIZED+DECENTRALIZED POLICY AND BROADCAST
                # if i > 0:
                #     agents_last_broadcast_embedding = copy.deepcopy(agents_broadcast_embedding)


                agents_action = []
                agents_broadcast = []

                agents_act_dist = []
                agents_values = []
                agents_values_b = []
                agents_memory = []
                agents_term_dist = []
                agents_broadcast_dist = []
                agents_embedding = []
                agents_broadcast_embedding = []


                total_broadcast_list = np.zeros(self.num_procs)

                for j, obs_j in enumerate(self.current_obss):

                    # Do one agent's forward propagation

                    preprocessed_obs = self.preprocess_obss(obs_j, device=self.device)

                    act_dist, values, values_b, memory, term_dist, broadcast_dist, embedding = \
                        self.acmodel.forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                                                          * self.current_mask.unsqueeze(1))

                    # collect outputs for each agent
                    agents_act_dist.append(act_dist)
                    agents_values.append(values)
                    agents_values_b.append(values_b)
                    agents_memory.append(memory)
                    agents_term_dist.append(term_dist)
                    agents_broadcast_dist.append(broadcast_dist)
                    agents_embedding.append(embedding)


                    # broadcast selection for each agent
                    if self.acmodel.use_broadcasting:

                        if self.always_broadcast:
                            broadcast = torch.ones((self.num_procs,), device=self.device)
                        else:
                            broadcast = broadcast_dist.sample()[range(self.num_procs), self.current_options[j].long()]
                        agents_broadcast.append(broadcast)
                        agents_broadcast_embedding.append(broadcast.unsqueeze(1).float() * embedding) # check embedding before and after multiplying with broadcast

                    # action selection

                    action = agents_act_dist[j].sample()[range(self.num_procs), self.current_options[j].long()]
                    agents_action.append(action)


                # compute option-action values with coordinator (doc)
                # FOR CENTRALIZED+DECENTRALIZED POLICY AND BROADCAST
                # if i == 0:
                #     agents_last_broadcast_embedding = copy.deepcopy(agents_embedding)

#                assert agents_values.count(None) == len(agents_values)
                if self.acmodel.use_central_critic:
                    coord_opt_act_values = torch.zeros((self.num_procs, self.num_options, self.num_actions, \
                                                        self.num_agents), device=self.device)

                    all_opt_act_values = torch.zeros((self.num_procs, self.num_options, self.num_actions, \
                                                        self.num_agents), device=self.device)
                    all_opt_act_values_b = torch.zeros((self.num_procs, self.num_options, 2, \
                                                      self.num_agents), device=self.device)
                    new_coord_memories = torch.zeros((self.num_procs, self.num_options, self.num_actions, \
                                                      self.acmodel.coord_memory_size), device=self.device)



                    for j in range(self.num_agents):
                        # FOR CENTRALIZED+DECENTRALIZED POLICY AND BROADCAST
                        # Replace broadcast embedding [j] with agents own embedding
                        # modified_agents_broadcast_embedding = copy.deepcopy(agents_broadcast_embedding)
                        # modified_agents_broadcast_embedding[j] = agents_embedding[j]

                        # modified_agents_last_broadcast_embedding = copy.deepcopy(agents_last_broadcast_embedding)
                        # modified_agents_last_broadcast_embedding[j] = agents_embedding[j]

                        for o in range(self.num_options):

                            for a in range(self.num_actions):
                                for b in range(2):
                                    # TODO: big bottleneck here. these loops are extremely inefficient

                                    option_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=o)
                                    action_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=a)
                                    broadcast_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=b)

                                    option_idxs = [option_idxs_agent_j if k == j else self.current_options[k] for k in range(self.num_agents)]
                                    action_idxs = [action_idxs_agent_j if k == j else agents_action[k] for k in range(self.num_agents)]
                                    broadcast_idxs = [broadcast_idxs_agent_j if k == j else agents_broadcast[k] for k in range(self.num_agents)]

                                    # TODO: because we need action here, action selection now happen before option selection. Make sure all the rest makes sense with that


                                    # mod_agent_value_b, _ = self.acmodel.forward_central_critic(
                                    #     modified_agents_last_broadcast_embedding,
                                    #     option_idxs,
                                    #     broadcast_idxs,
                                    #     self.current_coord_memory)



                                    # mod_agent_values, _ = self.acmodel.forward_central_critic(modified_agents_last_broadcast_embedding,
                                    #                                                             option_idxs,
                                    #                                                             action_idxs,
                                    #                                                             broadcast_idxs,
                                    #                                                             self.current_coord_memory* self.current_mask.unsqueeze(1))

                                    mod_agent_values, _ = self.acmodel.forward_central_critic(
                                        agents_broadcast_embedding,
                                        option_idxs,
                                        action_idxs,
                                        broadcast_idxs,
                                        self.current_coord_memory * self.current_mask.unsqueeze(1))

                                    mod_agent_values_b, _ = self.acmodel.forward_central_critic(
                                        agents_broadcast_embedding,
                                        option_idxs,
                                        action_idxs,
                                        broadcast_idxs,
                                        self.current_coord_memory * self.current_mask.unsqueeze(1))

                                    all_opt_act_values[:, o, a, j] = mod_agent_values #torch.tensor(np.array(mod_agent_values)[:,0])
                                    all_opt_act_values_b[:, o, b, j] = mod_agent_values_b #torch.tensor(np.array(mod_agent_values)[:,1])

                                    coord_value, coord_new_memory = self.acmodel.forward_central_critic(
                                        agents_broadcast_embedding,
                                        option_idxs,
                                        action_idxs,
                                        broadcast_idxs,
                                        self.current_coord_memory* self.current_mask.unsqueeze(1))
                                    for i, (op, ac) in enumerate(zip(option_idxs,action_idxs)):
                                        coord_opt_act_values[:, op.long(), ac.long(), i] = coord_value #torch.tensor(np.array(coord_values)[:,0])
                                        new_coord_memories[:, op.long(), ac.long(), :] = coord_new_memory


                        agents_values[j] = all_opt_act_values[:,:,:,j]
                        agents_values_b[j] = all_opt_act_values_b[:,:,:,j]



                        mean_coord_all_opt_act_values = torch.mean(coord_opt_act_values, dim=self.agt_dim, keepdim=True)


                for j in range(self.num_agents):
                    # Option-value
                    if self.acmodel.use_act_values:

                        # Compute joint action prob:
                        joint_act_prob = torch.ones_like(agents_act_dist[0].probs)

                        for j in range(self.num_agents):
                            joint_act_prob *= agents_act_dist[j].probs


                        # Required for option selection
                        Qsw_coord_all = torch.sum(joint_act_prob * mean_coord_all_opt_act_values.squeeze(),
                                                  dim=self.act_dim,
                                                  keepdim=True)
                        Qsw_coord_max, Qsw_coord_argmax = torch.max(Qsw_coord_all, dim=self.opt_dim, keepdim=True)
                        Qsw_coord= np.mean(torch.stack([Qsw_coord_all[range(self.num_procs), self.current_options[j].long()] for j in range(self.num_agents)]).tolist()) #coord_value
                        #can we use np.mean([Qsw_coord_all[range(self.num_procs), self.current_options[j].long()] for j in range(self.num_agents)]) instead of coord_value
                        Vs_coord = Qsw_coord_max

                        Qsw_all = torch.sum(agents_act_dist[j].probs * agents_values[j], dim=self.act_dim, keepdim=True)
                        Qsw_max, Qsw_argmax = torch.max(Qsw_all, dim=self.opt_dim, keepdim=True)
                        Qsw = Qsw_all[range(self.num_procs), self.current_options[j].long()]
                        Vs = Qsw_max


                        Qsw_b_all = torch.sum(agents_broadcast_dist[j].probs * agents_values_b[j], dim=self.act_dim,
                                            keepdim=True)
                        Qsw_b_max, Qsw_b_argmax = torch.max(Qsw_b_all, dim=self.opt_dim, keepdim=True)
                        Qsw_b = Qsw_b_all[range(self.num_procs), self.current_options[j].long()]
                        Vs_b = Qsw_b_max


                        # # Compute agents' critics
                        # Qswa = agents_values[j][range(self.num_procs), self.current_options[j].long(), agents_action[j]]
                        # Qswa_max, Qswa_argmax = torch.max(agents_values[j], dim=self.act_dim, keepdim=True)
                        #
                        # Vsw = Qswa_max[self.current_options[j].long()]


                        # # required for learning to broadcast
                        # Qswa_b = agents_values_b[j][range(self.num_procs), self.current_options[j].long(), agents_broadcast[j]]
                        # Qswa_b_max, Qswa_b_argmax = torch.max(agents_values_b[j], dim=self.act_dim, keepdim=True)
                        # Vsw_b = Qswa_b_max

                        self.rollout_coord_value_swa[i] = coord_value #torch.tensor(np.array(coord_values)[:,0])
                        self.rollout_coord_value_sw[i] = Qsw_coord.squeeze()
                        self.rollout_coord_value_s[i] = Vs_coord.squeeze()
                        self.rollout_coord_value_sw_max[i] = Qsw_coord_max.squeeze()





                    if self.acmodel.use_term_fn:

                        assert self.acmodel.use_act_values

                        # check termination

                        terminate = agents_term_dist[j].sample()[range(self.num_procs), self.current_options[j].long()]

                        # changes option (for those that terminate)

                        random_mask = terminate * (torch.rand(self.num_procs) < self.option_epsilon).float()
                        chosen_mask = terminate * (1. - random_mask)
                        assert all(torch.ones(self.num_procs) == random_mask + chosen_mask + (1. - terminate))

                        random_options = random_mask * torch.randint(self.num_options, size=(self.num_procs,)).float()
                        chosen_options = chosen_mask * Qsw_coord_argmax.squeeze().float()
                        self.current_options[j] = random_options + chosen_options + (1. - terminate) * self.current_options[j]

                    # update experience values (pre-step)

                    self.rollout_actions[j][i] = agents_action[j]
                    self.rollout_options[j][i] = self.current_options[j]
                    self.rollout_log_probs[j][i] = agents_act_dist[j].logits[range(self.num_procs), self.current_options[j].long(), agents_action[j]]

                    #self.rollout_masked_embeddings[j][i] = agents_broadcast_embedding[j]


                    if self.acmodel.recurrent:
                        self.rollout_agent_memories[j][i] = self.current_agent_memories[j]
                        self.current_agent_memories[j] = agents_memory[j]


                    if self.acmodel.use_act_values:
                        self.rollout_values_swa[j][i] = agents_values[j][
                            range(self.num_procs), self.current_options[j].long(), agents_action[j]].squeeze()

                        self.rollout_values_sw[j][i] = Qsw.squeeze()
                        self.rollout_values_s[j][i] = Vs.squeeze()
                        self.rollout_values_sw_max[j][i] = Qsw_max.squeeze()

                    else:
                        self.rollout_values[j][i] = agents_values[j][range(self.num_procs), self.current_options[j].long()].squeeze()
                        self.rollout_values_b[j][i] = agents_values_b[j][
                            range(self.num_procs), self.current_options[j].long()].squeeze()

                    if self.acmodel.use_term_fn:

                        self.rollout_terminates_prob[j][i] = agents_term_dist[j].probs[range(self.num_procs), self.current_options[j].long()]
                        self.rollout_terminates[j][i] = terminate

                    if self.acmodel.use_broadcasting:

                        # #self.rollout_broadcast_probs[j][i] = agents_broadcast_dist[j].probs[range(self.num_procs), self.current_options[j].long()]
                        self.rollout_broadcast_masks[j][i] = agents_broadcast[j]

                        # self.rollout_broadcast_log_probs[j][i] = agents_broadcast_dist[j].logits[range(self.num_procs), self.current_options[j].long(), agents_broadcast[j]]
                        # self.rollout_values_b[j][i] = Qswa_b.squeeze()
                        # self.rollout_values_sw_b[j][i] = Vsw_b[range(self.num_procs), self.current_options[j].long()].squeeze()

                        self.rollout_values_swa_b[j][i] = agents_values_b[j][
                            range(self.num_procs), self.current_options[j].long(), agents_broadcast[j]].squeeze()
                        self.rollout_values_sw_b[j][i] = Qsw_b.squeeze()
                        self.rollout_values_s_b[j][i] = Vs_b.squeeze()
                        self.rollout_values_sw_b_max[j][i] = Qsw_b_max.squeeze()


                    if self.acmodel.use_central_critic:

                        self.rollout_coord_memories[i] = self.current_coord_memory
                        # x = new_coord_memories[range(self.num_procs), self.current_options[j].long(), agents_action[j].long(), :]
                        x = coord_new_memory
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
                        a = torch.tensor(reward, device=self.device)

                        b = torch.tensor(agents_broadcast[j].unsqueeze(1).float()*self.broadcast_penalty, device=self.device)

                        self.rollout_rewards_plus_broadcast_penalties[j][i] = torch.add(a,b.squeeze().long())

                    if self.acmodel.use_term_fn:
                        # change current_option w.r.t. episode ending
                        self.current_options[j] = self.current_mask * self.current_options[j] + \
                                                  (1. - self.current_mask) * torch.randint(low=0, high=self.num_options,
                                                                                         size=(self.num_procs,),
                                                                                         device=self.device,
                                                                                         dtype=torch.float)

                    # update log values

                    self.log_episode_return[j] += self.rollout_rewards[j][i] #torch.tensor(reward, device=self.device, dtype=torch.float)
                    self.log_episode_return_with_broadcast_penalties[j] += self.rollout_rewards_plus_broadcast_penalties[j][i]
                    self.log_episode_reshaped_return[j] += self.rollout_rewards[j][i]
                    self.log_episode_num_frames[j] += torch.ones(self.num_procs, device=self.device)

                    for k, done_ in enumerate(done):
                        if done_:
                            self.log_done_counter[j] += 1
                            self.log_return[j].append(self.log_episode_return[j][k].item())
                            self.log_return_with_broadcast_penalties[j].append(self.log_episode_return_with_broadcast_penalties[j][k].item())
                            self.log_reshaped_return[j].append(self.log_episode_reshaped_return[j][k].item())
                            self.log_num_frames[j].append(self.log_episode_num_frames[j][k].item())

                    self.log_episode_return[j] *= self.current_mask
                    self.log_episode_return_with_broadcast_penalties[j] *= self.current_mask
                    self.log_episode_reshaped_return[j] *= self.current_mask
                    self.log_episode_num_frames[j] *= self.current_mask

            # Add advantage and return to experiences

            for j in range(self.num_agents):

                for i in reversed(range(self.num_frames_per_proc)):
                    if self.acmodel.use_term_fn and self.acmodel.use_act_values:
                        #next_mask = self.rollout_masks[i + 1]

                        # For coordinator  q-learning (centralized)
                        agents_no_term_prob_array = np.array([1. - self.rollout_terminates_prob[j][i + 1].numpy() for j in range(self.num_agents)])
                        no_agent_term_prob = agents_no_term_prob_array.prod()



                        # the env-reward is copied for each agent, so we can take for any j self.rollout_rewards[j][i]
                        interim = torch.stack(agents_broadcast)
                        total_broadcast_list = np.sum(np.array(interim.tolist()), 0)

                        self.rollout_coord_target[i] = self.rollout_rewards[0][i] - torch.tensor(total_broadcast_list*self.broadcast_penalty, device=self.device).float() + \
                                         self.rollout_masks[i + 1] * self.discount * \
                                         (
                                                 no_agent_term_prob * self.rollout_coord_value_sw[i+1] + \
                                                 (1. - no_agent_term_prob) * self.rollout_coord_value_sw_max[i+1]
                                         )

                        self.rollout_coord_advantages[i] = self.rollout_coord_value_sw[i + 1] - self.rollout_coord_value_s[
                            i + 1]


                        # q-learning for each agent's policy (decentralized)
                        # next_value = self.rollout_values_swa[j][i + 1]
                        # next_advantage = self.rollout_advantages[j][i + 1] if i < self.num_frames_per_proc else 0
                        #
                        # self.rollout_targets[j][i] = self.rollout_rewards[j][i] + self.discount * next_value * next_mask
                        #
                        # self.rollout_advantages[j][i] = self.rollout_targets[j][i]  - self.rollout_values_swa[j][i] + \
                        #                                 self.discount * self.gae_lambda * next_advantage * next_mask

                        self.rollout_targets[j][i] = self.rollout_rewards[j][i] + \
                                                     self.rollout_masks[i + 1] * self.discount * \
                                                     (
                                                             (1. - self.rollout_terminates_prob[j][i + 1]) *
                                                             self.rollout_values_sw[j][i + 1] + \
                                                             self.rollout_terminates_prob[j][i + 1] *
                                                             self.rollout_values_sw_max[j][i + 1]
                                                     )

                        # option-advantage for action

                        self.rollout_advantages[j][i] = self.rollout_values_sw[j][i + 1] - self.rollout_values_s[j][
                            i + 1]

                        # q-learning for each agent's broadcast (decentralized): 'b' denotes broadcast

                        self.rollout_targets_b[j][i] = self.rollout_rewards_plus_broadcast_penalties[j][i] + \
                                                       self.rollout_masks[i + 1] * self.discount * \
                                                       (
                                                               (1. - self.rollout_terminates_prob[j][i + 1]) *
                                                               self.rollout_values_sw_b[j][i + 1] + \
                                                               self.rollout_terminates_prob[j][i + 1] *
                                                               self.rollout_values_sw_b_max[j][i + 1]
                                                       )

                        # option-advantage for broadcast

                        self.rollout_advantages_b[j][i] = self.rollout_values_sw_b[j][i + 1] - self.rollout_values_s_b[j][
                            i + 1]

                        next_value_b = self.rollout_values_swa_b[j][i + 1]
                        next_advantage_b = self.rollout_advantages[j][i + 1] if i < self.num_frames_per_proc else 0

                        self.rollout_targets_b[j][i] = self.rollout_rewards_plus_broadcast_penalties[j][i] + self.discount * next_value_b * self.rollout_masks[i + 1]
                        self.rollout_advantages_b[j][i] = self.rollout_targets_b[j][i] - self.rollout_values_swa_b[j][i]+ \
                                                        self.discount * self.gae_lambda * next_advantage_b * self.rollout_masks[i + 1]

                    elif not self.acmodel.use_term_fn and not self.acmodel.use_act_values:
                        next_mask = self.rollout_masks[i+1]
                        next_value = self.rollout_values[j][i+1]
                        next_value_b = self.rollout_values_b[i+1]
                        next_advantage = self.rollout_advantages[j][i+1] if i < self.num_frames_per_proc else 0
                        next_advantage_b = self.rollout_advantages_b[j][i + 1] if i < self.num_frames_per_proc else 0

                        delta = self.rollout_rewards[j][i] + self.discount * next_value * self.rollout_masks[i + 1] - self.rollout_values[j][i]
                        self.rollout_advantages[j][i] = delta + self.discount * self.gae_lambda * next_advantage * self.rollout_masks[i + 1]



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
            coord_exps = DictList()
            logs = {k:[] for k in ["return_per_episode",
                                   "return_per_episode_with_broadcast_penalties",
                                   "reshaped_return_per_episode",
                                   "num_frames_per_episode",
                                   "num_frames",
                                   "entropy",
                                   "broadcast_entropy"
                                   "value",
                                   "policy_loss",
                                   "value_loss",
                                   "grad_norm"]}


           #  import pdb;
           #  pdb.set_trace()
            for j in range(self.num_agents):
                exps[j].obs = [self.rollout_obss[i][j][k] for k in range(self.num_procs) \
                               for i in range(self.num_frames_per_proc)]

                if self.acmodel.recurrent:
                    # T x P x D -> P x T x D -> (P * T) x D
                    coord_exps.memory = self.rollout_coord_memories[:-1].transpose(0,1).reshape(-1, *self.rollout_coord_memories.shape[2:])
                    exps[j].memory = self.rollout_agent_memories[j][:-1].transpose(0, 1).reshape(-1, *self.rollout_agent_memories[j].shape[2:])
                    # T x P -> P x T -> (P * T) x 1
                    exps[j].mask = self.rollout_masks[:-1].transpose(0, 1).reshape(-1).unsqueeze(1)
                    # TODO: Fix the shape of exps[j].last_masked_embedding
                    # exps[j].last_masked_embedding = self.rollout_masked_embeddings[j][-2]


                # for all tensors below, T x P -> P x T -> P * T
                exps[j].action = self.rollout_actions[j][:-1].transpose(0, 1).reshape(-1) #.long()
                exps[j].broadcast = self.rollout_broadcast_masks[j][:-1].transpose(0, 1).reshape(-1) #.long()

                #exps[j].last_broadcast = self.rollout_broadcast_masks[j][:-2].transpose(0, 1).reshape(-1).long()


                exps[j].reward = self.rollout_rewards[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].reward_b = self.rollout_rewards_plus_broadcast_penalties[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].advantage = self.rollout_advantages[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].advantage_b = self.rollout_advantages_b[j][:-1].transpose(0, 1).reshape(-1)

                exps[j].log_prob = self.rollout_log_probs[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].broadcast_log_prob = self.rollout_broadcast_log_probs[j][:-1].transpose(0, 1).reshape(-1)
                exps[j].current_options = self.rollout_options[j][:-1].transpose(0, 1).reshape(-1).long()

                if self.acmodel.use_term_fn:

                    exps[j].terminate = self.rollout_terminates[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].terminate_prob = self.rollout_terminates_prob[j][:-1].transpose(0, 1).reshape(-1)


                if self.acmodel.use_act_values:
                    coord_exps.value_swa = self.rollout_coord_value_swa[:-1].transpose(0, 1).reshape(-1)
                    coord_exps.value_sw = self.rollout_coord_value_sw[:-1].transpose(0, 1).reshape(-1)
                    coord_exps.value_s = self.rollout_coord_value_s[:-1].transpose(0, 1).reshape(-1)
                    coord_exps.advantage = self.rollout_coord_advantages[:-1].transpose(0, 1).reshape(-1)


                    exps[j].value_swa = self.rollout_values_swa[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].value_sw = self.rollout_values_sw[j][:-1].transpose(0, 1).reshape(-1)

                    exps[j].value_swa_b = self.rollout_values_swa_b[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].value_sw_b = self.rollout_values_sw_b[j][:-1].transpose(0, 1).reshape(-1)

                    exps[j].target = self.rollout_targets[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].target_b = self.rollout_targets_b[j][:-1].transpose(0, 1).reshape(-1)

                    coord_exps.target = self.rollout_coord_target[:-1].transpose(0, 1).reshape(-1)


                else:
                    exps[j].value = self.rollout_values[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].value_b = self.rollout_values_b[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].returnn = exps[j].value + exps[j].advantage

                # Preprocess experiences

                exps[j].obs = self.preprocess_obss(exps[j].obs, device=self.device)

                # Log some values

                keep = max(self.log_done_counter[j], self.num_procs)

                logs["return_per_episode"].append(self.log_return[j][-keep:])
                logs["return_per_episode_with_broadcast_penalties"].append(self.log_return_with_broadcast_penalties[j][-keep:])
                logs["reshaped_return_per_episode"].append(self.log_reshaped_return[j][-keep:])
                logs["num_frames_per_episode"].append(self.log_num_frames[j][-keep:])
                logs["num_frames"].append(self.num_frames)

                self.log_return[j] = self.log_return[j][-self.num_procs:]
                self.log_return_with_broadcast_penalties[j] = self.log_return_with_broadcast_penalties[j][-self.num_procs:]
                self.log_reshaped_return[j] = self.log_reshaped_return[j][-self.num_procs:]
                self.log_num_frames[j] = self.log_num_frames[j][-self.num_procs:]

        return coord_exps, exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
