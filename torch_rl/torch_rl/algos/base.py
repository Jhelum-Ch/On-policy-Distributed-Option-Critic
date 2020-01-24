from abc import ABC, abstractmethod
import torch
import numpy as np
import itertools
import copy
from collections import defaultdict


from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv


# The following two methods are copied from MADDPG train.py. Modify in pytorch to suit our case
# def make_env(scenario_name, benchmark=False):
#     from multiagent.environment import MultiAgentEnv
#     import multiagent.scenarios as scenarios
#
#     # load scenario from script
#     scenario = scenarios.load(scenario_name + ".py").Scenario()
#     # create world
#     world = scenario.make_world()
#     # create multiagent environment
#     if benchmark:
#         env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
#     else:
#         env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
#     return env

def esimate_embedding(masked_embedding, broadcast, rollout_embedding, broadcast_idx):
    est_embed = copy.deepcopy(masked_embedding)
    for k in range(len(broadcast)):
        if broadcast[k] == 1:
            est_embed[k] == masked_embedding[k]
        else:
            est_embed[k] = rollout_embedding[broadcast_idx[k].long()][k]
    return est_embed



class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, num_agents=None, envs=None, acmodel=None, replay_buffer=None, \
                 num_frames_per_proc=None, discount=None, lr=None, gae_lambda=None, \
                 entropy_coef=None,
                 value_loss_coef=None, recurrence=None, \
                 preprocess_obss=None, reshape_reward=None, max_grad_norm=None, broadcast_penalty=-0.01,
                 termination_reg=0.01, termination_loss_coef=0.5, option_epsilon=0.05
                 ):
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
        #self.max_len_ep = max_len_ep
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
        self.replay_buffer = replay_buffer
        self.always_broadcast = self.acmodel.always_broadcast
        self.broadcast_penalty = broadcast_penalty if not self.always_broadcast else 0.
        self.num_actions = self.acmodel.num_actions
        #self.num_options = num_options
        self.num_options = self.acmodel.num_options
        self.term_loss_coef = termination_loss_coef
        self.termination_reg = termination_reg
        self.option_epsilon = option_epsilon
        self.num_broadcasts = 1 if self.acmodel.always_broadcast or not self.acmodel.recurrent else 2
        #self.scenario = scenario


        #print('max_len_ep', self.acmodel.max_len_ep)

        # Dimension convention

        self.batch_dim = 0
        self.opt_dim = 1
        self.act_dim = 2
        self.brd_dim = 3
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

        # if not self.acmodel.recurrent:
        #     self.num_broadcasts = 1
        # else :
        #     raise NotImplemented

        # Initialize experience values

        self.shape = (self.num_frames_per_proc + 1, self.num_procs)

        self.current_obss = self.env.reset()
        self.rollout_obss = [None]*(self.shape[0])
        self.rollout_next_obss = [None] * (self.shape[0])


        if self.acmodel.recurrent:

            if self.acmodel.use_teamgrid:
                self.current_agent_memories = [torch.zeros(self.shape[1], self.acmodel.memory_size, device=self.device) for _ in range(self.num_agents)]
                self.rollout_agent_memories = [torch.zeros(*self.shape, self.acmodel.memory_size, device=self.device) for _ in range(self.num_agents)]
                self.rollout_agent_embeddings = [torch.zeros(*self.shape, self.acmodel.semi_memory_size, device=self.device) for _ in range(self.num_agents)]
            else:
                self.current_agent_memories = [torch.zeros(self.shape[1], self.acmodel.memory_size[j], device=self.device) for j
                                               in range(self.num_agents)]
                self.rollout_agent_memories = [torch.zeros(*self.shape, self.acmodel.memory_size[j], device=self.device) for j
                                               in range(self.num_agents)]
                self.rollout_agent_embeddings = [torch.zeros(*self.shape, self.acmodel.semi_memory_size[j], device=self.device)
                                                 for j in range(self.num_agents)]

            # if not self.acmodel.use_teamgrid:
            #     self.particle_agent_memories = [torch.zeros(self.shape[1], self.acmodel.memory_size_for_particle[j],
            #                                             device=self.device) for j in range(self.num_agents)]
            #     self.rollout_particle_agent_memories = [torch.zeros(*self.shape, self.acmodel.memory_size_for_particle[j],
            #                                               device=self.device) for j in range(self.num_agents)]

            if self.acmodel.use_central_critic:
                self.current_coord_memory = torch.zeros(self.shape[1], self.acmodel.coord_memory_size, device=self.device)
                self.rollout_coord_memories = torch.zeros(*self.shape, self.acmodel.coord_memory_size, device=self.device)



        self.current_mask = torch.ones(self.shape[1], device=self.device)
        self.rollout_masks = torch.zeros(*self.shape, device=self.device)

        if self.acmodel.use_teamgrid:
            self.rollout_masked_embeddings = [torch.zeros(*self.shape, self.acmodel.semi_memory_size, device=self.device) for _ in range(self.num_agents)]
            self.rollout_estimated_embeddings = [torch.zeros(*self.shape, self.acmodel.semi_memory_size, device=self.device) for _
                                              in range(self.num_agents)]
            self.rollout_next_estimated_embeddings = [
                torch.zeros(*self.shape, self.acmodel.semi_memory_size, device=self.device) for _
                in range(self.num_agents)]
            self.rollout_actions_mlp = [torch.zeros(*self.shape, self.num_actions, device=self.device, dtype=torch.int) for _ in
                                        range(self.num_agents)] #for current option
        else:
            self.rollout_masked_embeddings = [torch.zeros(*self.shape, self.acmodel.semi_memory_size[j], device=self.device) for
                                              j in range(self.num_agents)]
            self.rollout_estimated_embeddings = [torch.zeros(*self.shape, self.acmodel.semi_memory_size[j], device=self.device)
                                                 for j
                                                 in range(self.num_agents)]
            self.rollout_next_estimated_embeddings = [
                torch.zeros(*self.shape, self.acmodel.semi_memory_size[j], device=self.device)
                for j
                in range(self.num_agents)]
            # self.rollout_particle_agent_embeddings = [torch.zeros(*self.shape, self.acmodel.semi_memory_size_for_particle[j], device=self.device) \
            #                                           for j in range(self.num_agents)]
            self.rollout_actions_mlp = [torch.zeros(*self.shape, self.num_actions[j], device=self.device, dtype=torch.int) for j in
                                        range(self.num_agents)] #for current option

        self.rollout_actions = [torch.zeros(*self.shape, device=self.device, dtype=torch.int) for _ in range(self.num_agents)]

        self.rollout_rewards = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_rewards_plus_broadcast_penalties = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_advantages = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_advantages_b = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_log_probs = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_broadcast_log_probs = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]

        self.rollout_options = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]

        if self.acmodel.use_term_fn:

            self.rollout_terminates = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_terminates_prob = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]

        if self.num_options is not None:
            self.current_options = [torch.randint(low=0, high=self.num_options, size=(self.shape[1],), device=self.device, dtype=torch.float) for _ in range(self.num_agents)]
           # self.current_joint_option = [torch.zeros(shape[1], device=self.device) for _ in range(self.num_agents)]

        else:

            self.current_options = torch.arange(self.num_agents)

        self.current_joint_action = [torch.zeros(self.shape[1], device=self.device) for _ in range(self.num_agents)]

        #print('self.acmodel.use_act_values', self.acmodel.use_act_values)
        if self.acmodel.use_act_values:

            # if self.acmodel.use_central_critic:
            #     self.rollout_coord_value_swa = torch.zeros(*shape, device=self.device)
            #     self.rollout_coord_value_sw = torch.zeros(*shape, device=self.device)
            #     self.rollout_coord_value_s = torch.zeros(*shape, device=self.device)
            #     self.rollout_coord_value_sw_max = torch.zeros(*shape, device=self.device)
            #
            #     self.rollout_coord_target = torch.zeros(*shape, device=self.device)


            self.rollout_values_swa = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_sw = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]


            self.rollout_values_s = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
            self.rollout_values_sw_max = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]

            self.rollout_targets = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
            #self.rollout_targets_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]


        else:

            self.rollout_values = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
            #self.rollout_values_b = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]



        # replay buffer for all agents
        #self.buffer = [[] for _ in range(self.num_agents)]
        # self.rollout_buffer = [None for i in range(self.shape[0]) for j in range(self.num_agents)]
        self.rollout_buffer = [[None for i in range(self.shape[0])] for j in range(self.num_agents)]
       # if self.acmodel.use_broadcasting:

        self.current_broadcast_state = [torch.ones(self.shape[1], device=self.device) for _ in range(self.num_agents)]

        self.rollout_values_swa_b = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_values_sw_b = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_values_s_b = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_values_sw_b_max = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        self.rollout_advantages_b = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]

        self.rollout_broadcast_masks = [torch.zeros(*self.shape, device=self.device) for _ in range(self.num_agents)]
        #self.rollout_broadcast_probs = [torch.zeros(*shape, device=self.device) for _ in range(self.num_agents)]


        # Initialize log values

        self.log_episode_return = [torch.zeros(self.shape[1], device=self.device) for _ in range(self.num_agents)]
        self.log_episode_return_with_broadcast_penalties = [torch.zeros(self.shape[1], device=self.device) for _ in range(self.num_agents)]
        self.log_episode_reshaped_return = [torch.zeros(self.shape[1], device=self.device) for _ in range(self.num_agents)]
        self.log_episode_num_frames = [torch.zeros(self.shape[1], device=self.device) for _ in range(self.num_agents)]

        self.log_done_counter = 0
        self.env_step = 0
        self.log_return = [[0] * self.shape[1] for _ in range(self.num_agents)]
        self.log_return_with_broadcast_penalties = [[0] * self.shape[1] for _ in range(self.num_agents)]
        self.log_mean_agent_return = [0] * self.shape[1]
        self.log_reshaped_return = [[0] * self.shape[1] for _ in range(self.num_agents)]
        self.log_num_frames = [[0] * self.shape[1] for _ in range(self.num_agents)]

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
        #print('episode_return', self.log_episode_return_with_broadcast_penalties)
        if not self.recurrence:
            raise Exception("Deprecated: self.recurrence has to be True."
                            "If no reccurence is used, we will still have self.recurrence=True"
                            "but self.acmodel.use_memory_agents will be set to False.")

        with torch.no_grad():

            self.rollout_length = len(self.rollout_obss)

            #agents_broadcast_idx = [0 for _ in range(self.num_agents)]
            agents_broadcast_idx = torch.zeros((self.num_agents, self.num_procs), device=self.device)

            for i in range(self.rollout_length):

                #print('i', i)
                self.rollout_obss[i] = self.current_obss


                # FOR CENTRALIZED+DECENTRALIZED POLICY AND BROADCAST
                # if i > 0:
                #     agents_last_broadcast_embedding = copy.deepcopy(agents_broadcast_embedding)


                agents_action = []
                agents_broadcast = []

                agents_act_dist = []
                agents_action_mlp = []
                agents_values = []
                agents_values_b = []
                agents_memory = []
                agents_term_dist = []
                agents_broadcast_dist = []
                agents_embedding = []
                particle_agents_embedding = []
                particle_agents_memory = []
                agents_broadcast_embedding = []
                agents_estimated_embedding = []
                agents_next_estimated_embedding = []



                total_broadcast_list = np.zeros(self.num_procs)


                for j, obs_j in enumerate(self.current_obss):
                    #import ipdb; ipdb.set_trace()
                    # Do one agent's forward propagation
                    #print('LEN', np.shape(self.current_obss[0]))
                    preprocessed_obs = self.preprocess_obss(obs_j, device=self.device)

                    #if self.acmodel.use_teamgrid:
                    if self.num_options is not None:
                        if not self.acmodel.always_broadcast:
                            act_mlp, act_dist, values, values_b, memory, term_dist, broadcast_dist, embedding = \
                                self.acmodel.forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                                                                  * self.current_mask.unsqueeze(1), agent_index = j)

                            agents_values_b.append(values_b)
                            agents_broadcast_dist.append(broadcast_dist)

                        else:
                            act_mlp, act_dist, values, memory, term_dist, embedding = \
                                self.acmodel.forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                                                                  * self.current_mask.unsqueeze(1), agent_index = j)



                    else:
                        if not self.acmodel.always_broadcast:
                            act_mlp, act_dist, values, values_b, memory, broadcast_dist, embedding = \
                                self.acmodel.forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                                                                  * self.current_mask.unsqueeze(1), agent_index = j)
                            agents_values_b.append(values_b)
                            agents_broadcast_dist.append(broadcast_dist)
                        else:
                            act_mlp, act_dist, values, memory, embedding = \
                                self.acmodel.forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                                                                  * self.current_mask.unsqueeze(1), agent_index = j)
                    # else:
                    #     if self.num_options is not None:
                    #         if not self.acmodel[j].always_broadcast:
                    #             act_dist, values, values_b, memory, term_dist, broadcast_dist, embedding = \
                    #                 self.acmodel[j].forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                    #                                                   * self.current_mask.unsqueeze(1), agent_index=j)
                    #
                    #             agents_values_b.append(values_b)
                    #             agents_broadcast_dist.append(broadcast_dist)
                    #
                    #         else:
                    #             # print('obs', preprocessed_obs, 'memory', self.current_agent_memories[j].size())
                    #             act_dist, values, memory, term_dist, embedding = \
                    #                 self.acmodel[j].forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                    #                                                   * self.current_mask.unsqueeze(1), agent_index=j)
                    #
                    #
                    #
                    #     else:
                    #         if self.acmodel[j].use_broadcasting and not self.acmodel[j].always_broadcast:
                    #             act_dist, values, values_b, memory, broadcast_dist, embedding = \
                    #                 self.acmodel[j].forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                    #                                                   * self.current_mask.unsqueeze(1), agent_index=j)
                    #             agents_values_b.append(values_b)
                    #             agents_broadcast_dist.append(broadcast_dist)
                    #         else:
                    #             act_dist, values, memory, embedding = \
                    #                 self.acmodel[j].forward_agent_critic(preprocessed_obs, self.current_agent_memories[j] \
                    #                                                   * self.current_mask.unsqueeze(1), agent_index=j)



                    action_mlp = act_mlp[range(self.num_procs), self.current_options[j].long()]
                    agents_action_mlp.append(action_mlp)

                    # collect outputs for each agent
                    agents_act_dist.append(act_dist)
                    agents_values.append(values)

                    agents_memory.append(memory)
                    agents_term_dist.append(term_dist)

                    agents_embedding.append(embedding)


                    # action selection

                    action = agents_act_dist[j].sample()[range(self.num_procs), self.current_options[j].long()]

                    agents_action.append(action)
                    #print('agents_action[j].long()', agents_action[j].long().size())
                    #print('j',j, 'agents_action_size', torch.stack(agents_action).size())

                    # broadcast selection for each agent
                    if i == 0:
                        broadcast = torch.ones((self.num_procs,), device=self.device) # We assume that each agent broadcasts at the first instant
                    else:
                        if self.acmodel.always_broadcast:
                            broadcast = torch.ones((self.num_procs,), device=self.device)
                        else:
                            broadcast = agents_broadcast_dist[j].sample()[range(self.num_procs), self.current_options[j].long(), agents_action[j].long()]


                    agents_broadcast.append(broadcast)
                    #agents_broadcast.append(broadcast)
                    agents_broadcast_embedding.append(broadcast.unsqueeze(
                        1).float() * embedding)  # check embedding before and after multiplying with broadcast


                    estimated_embedding = esimate_embedding(agents_broadcast_embedding[j], agents_broadcast[j], self.rollout_agent_embeddings[j], agents_broadcast_idx[j])
                    agents_estimated_embedding.append(estimated_embedding)



                # compute option-action values with coordinator (doc)
                # FOR CENTRALIZED+DECENTRALIZED POLICY AND BROADCAST
                # if i == 0:
                #     agents_last_broadcast_embedding = copy.deepcopy(agents_embedding)

#                assert agents_values.count(None) == len(agents_values)


                if self.num_options is not None:
                    #coord_opt_act_values = torch.zeros((self.num_procs, self.num_options, self.num_actions, \
                                                        # self.num_agents), device=self.device)
                    if self.acmodel.use_teamgrid:
                        all_opt_act_values = torch.zeros((self.num_procs, self.num_options, self.num_actions, \
                                                            self.num_broadcasts, self.num_agents), device=self.device) #
                        all_opt_act_target_values = torch.zeros((self.num_procs, self.num_options, self.num_actions, \
                                                        self.num_broadcasts, self.num_agents), device=self.device)

                        for j in range(self.num_agents):
                            for o in range(self.num_options):

                                for a in range(self.num_actions):
                                    for b in range(self.num_broadcasts):

                                        # TODO: big bottleneck here. these loops are extremely inefficient

                                        option_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=o)
                                        action_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=a)
                                        broadcast_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=b)
                                        #print('broadcast_idxs_agent_j', broadcast_idxs_agent_j)

                                        option_idxs = [option_idxs_agent_j if k == j else self.current_options[k] for k in range(self.num_agents)]
                                        #print('action_idxs_agent_j', action_idxs_agent_j, 'agents_action_size', torch.stack(agents_action).size())
                                        action_idxs = [action_idxs_agent_j if k == j else agents_action[k] for k in range(self.num_agents)]

                                        if self.acmodel.always_broadcast:
                                            broadcast_idxs = [agents_broadcast[k] for k in range(self.num_agents)]
                                        else:
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
                                        if self.acmodel.use_central_critic:
                                            _, mod_agent_values, new_coord_memory = self.acmodel.forward_central_critic(
                                                agents_broadcast_embedding,
                                                option_idxs,
                                                action_idxs,
                                                broadcast_idxs,
                                                self.current_coord_memory * self.current_mask.unsqueeze(1))

                                            #print('coord_embedding', coord_embedding)
                                            all_opt_act_values[:, o, a, b, j] = mod_agent_values #torch.tensor(np.array(mod_agent_values)[:,0])

                            agents_values[j] = all_opt_act_values[:,:,:,:,j]
                    else: #particle
                        all_opt_act_values = [torch.zeros(
                            (self.num_procs, self.num_options, self.num_actions[j], \
                                                                   self.num_broadcasts),
                                                                  device=self.device) for j in range(self.num_agents)]  #
                        all_opt_act_target_values = [torch.zeros(
                             (self.num_procs, self.num_options, self.num_actions[j], \
                              self.num_broadcasts),
                             device=self.device) for j in range(self.num_agents)]

                        for j in range(self.num_agents):
                            #print('j',j,'range(self.num_actions[j]', range(self.num_actions[j]))
                            for o in range(self.num_options):

                                for a in range(self.num_actions[j]):
                                    for b in range(self.num_broadcasts):

                                        # TODO: big bottleneck here. these loops are extremely inefficient

                                        option_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=o)
                                        action_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=a)
                                        broadcast_idxs_agent_j = torch.full(size=(self.num_procs,), fill_value=b)
                                        # print('broadcast_idxs_agent_j', broadcast_idxs_agent_j)

                                        option_idxs = [option_idxs_agent_j if k == j else self.current_options[k] for k
                                                       in range(self.num_agents)]
                                        # print('action_idxs_agent_j', action_idxs_agent_j, 'agents_action_size', torch.stack(agents_action).size())
                                        action_idxs = [action_idxs_agent_j if k == j else agents_action[k] for k in
                                                       range(self.num_agents)]

                                        if self.acmodel.always_broadcast:
                                            broadcast_idxs = [agents_broadcast[k] for k in range(self.num_agents)]
                                        else:
                                            broadcast_idxs = [broadcast_idxs_agent_j if k == j else agents_broadcast[k]
                                                              for k in range(self.num_agents)]

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
                                        if self.acmodel.use_central_critic:
                                        #print('base_ac_idx', action_idxs)
                                            _, mod_agent_values, new_coord_memory = self.acmodel.forward_central_critic(
                                                agents_broadcast_embedding,
                                                option_idxs,
                                                action_idxs,
                                                broadcast_idxs,
                                                self.current_coord_memory * self.current_mask.unsqueeze(1))

                                            # print('coord_embedding', coord_embedding)
                                            all_opt_act_values[j][:, o, a, b] = mod_agent_values  # torch.tensor(np.array(mod_agent_values)[:,0])
                                        # else:
                                        #     particle_agent_embedding, mod_agent_value, particle_agent_memory = \
                                        #         self.acmodel.forward_central_critic_others(agents_embedding[j], option_idxs, action_idxs,\
                                        #                                                    broadcast_idxs, self.particle_agent_memories[j],agent_index=j)
                                        #     print('mod', mod_agent_value.size())
                                        #     all_opt_act_values[j][:, o, a, b] = mod_agent_value
                                        #     particle_agents_embedding.append(particle_agent_embedding)
                                        #     particle_agents_memory.append(particle_agent_memory)

                            agents_values[j] = all_opt_act_values[j]




                        #agents_estimated_embedding[j] = coord_embedding[j]

                        #agents_target_values[j] = all_opt_act_target_values[:,:,:,j]
                        #agents_values_b[j] = all_opt_act_values_b[:,:,:,j]



                       # mean_coord_all_opt_act_values = torch.mean(coord_opt_act_values, dim=self.agt_dim, keepdim=True)


                for j in range(self.num_agents):
                    # Option-value
                    if self.acmodel.use_act_values:

                        if self.acmodel.use_teamgrid:
                            if self.acmodel.use_broadcasting and not self.acmodel.always_broadcast:
                                Q_avgd_brd = torch.sum(agents_broadcast_dist[j].probs * agents_values[j], dim=self.brd_dim, keepdim=True)
                                #print('dimQ', Q_avgd_brd.size())
                                Qsw_all = torch.sum(agents_act_dist[j].probs * Q_avgd_brd.squeeze(), dim=self.act_dim, keepdim=True)
                                # Qsw_all = torch.sum(agents_act_dist[j].probs * agents_values[j], dim=self.act_dim, keepdim=True)
                                Qsw_max, Qsw_argmax = torch.max(Qsw_all, dim=self.opt_dim, keepdim=True)
                                Qsw = Qsw_all[range(self.num_procs), self.current_options[j].long()]
                                Vs = Qsw_max
                            else:
                                Q_avgd_brd = torch.mean(agents_values[j], dim=self.brd_dim,
                                                       keepdim=True)
                                Qsw_all = torch.sum(agents_act_dist[j].probs * Q_avgd_brd.squeeze(), dim=self.act_dim,
                                                    keepdim=True)
                                # Qsw_all = torch.sum(agents_act_dist[j].probs * agents_values[j], dim=self.act_dim, keepdim=True)
                                Qsw_max, Qsw_argmax = torch.max(Qsw_all, dim=self.opt_dim, keepdim=True)
                                Qsw = Qsw_all[range(self.num_procs), self.current_options[j].long()]
                                Vs = Qsw_max
                        else:
                            if self.acmodel.use_broadcasting and not self.acmodel.always_broadcast:
                                Q_avgd_brd = torch.sum(agents_broadcast_dist[j].probs * agents_values[j], dim=self.brd_dim, keepdim=True)
                                # print('dimQ', Q_avgd_brd.size())
                                Qsw_all = torch.sum(agents_act_dist[j].probs * Q_avgd_brd.squeeze(), dim=self.act_dim, keepdim=True)
                                # Qsw_all = torch.sum(agents_act_dist[j].probs * agents_values[j], dim=self.act_dim, keepdim=True)
                                Qsw_max, Qsw_argmax = torch.max(Qsw_all, dim=self.opt_dim, keepdim=True)
                                Qsw = Qsw_all[range(self.num_procs), self.current_options[j].long()]
                                Vs = Qsw_max
                            else:
                                Q_avgd_brd = torch.mean(agents_values[j], dim=self.brd_dim,
                                                       keepdim=True)
                                #print('dimQ', Q_avgd_brd.size())
                                Qsw_all = torch.sum(agents_act_dist[j].probs * Q_avgd_brd.squeeze(), dim=self.act_dim,
                                                    keepdim=True)
                                # Qsw_all = torch.sum(agents_act_dist[j].probs * agents_values[j], dim=self.act_dim, keepdim=True)
                                Qsw_max, Qsw_argmax = torch.max(Qsw_all, dim=self.opt_dim, keepdim=True)
                                Qsw = Qsw_all[range(self.num_procs), self.current_options[j].long()]
                                Vs = Qsw_max



                    if self.acmodel.use_term_fn:

                        assert self.acmodel.use_act_values

                        # check termination

                        terminate = agents_term_dist[j].sample()[range(self.num_procs), self.current_options[j].long()]

                        # changes option (for those that terminate)

                        random_mask = terminate * (torch.rand(self.num_procs) < self.option_epsilon).float()
                        chosen_mask = terminate * (1. - random_mask)
                        assert all(torch.ones(self.num_procs) == random_mask + chosen_mask + (1. - terminate))

                        random_options = random_mask * torch.randint(self.num_options, size=(self.num_procs,)).float()
                        # chosen_options = chosen_mask * Qsw_coord_argmax.squeeze().float()
                        chosen_options = chosen_mask * Qsw_argmax.squeeze().float()
                        self.current_options[j] = random_options + chosen_options + (1. - terminate) * self.current_options[j]
                        #print('current_options[j]', self.current_options[j])
                    # update experience values (pre-step)



                    self.rollout_agent_embeddings[j][i] = agents_embedding[j] #embedding
                    # if not self.acmodel.use_teamgrid and not self.acmodel.use_central_critic:
                    #     self.rollout_particle_agent_embeddings[j][i] = particle_agents_embedding[j]  # embedding


                    #agent_broadcast_embedding_list = torch.zeros((self.num_procs,), self.acmodel.memory_size, device=self.device)

                    # for k in range(len(agents_broadcast[j])):
                    #     if agents_broadcast[j][k] == torch.tensor(1):
                    #         self.rollout_estimated_embeddings[j][i] = embedding
                    #     else:
                    #         agent_broadcast_embedding_list.append
                    #         #last_idx = agents_broadcast_idx[j][-1]
                    #         self.rollout_estimated_embeddings[j][i] = self.rollout_agent_embeddings[j][agents_broadcast_idx[j]]

                    self.rollout_estimated_embeddings[j][i] = agents_estimated_embedding[j]
                    #self.rollout_next_estimated_embeddings[j][i] =

                    # store broadcast indices
                    for k in range(len(broadcast)):
                        if broadcast[k] == 1:
                            agents_broadcast_idx[j][k] = i

                    self.rollout_actions[j][i] = agents_action[j]
                    self.rollout_actions_mlp[j][i] = agents_action_mlp[j]
                    self.rollout_options[j][i] = self.current_options[j]
                    self.rollout_log_probs[j][i] = agents_act_dist[j].logits[range(self.num_procs), self.current_options[j].long(), agents_action[j]]
                    #print('logit', agents_broadcast_dist[j].logits.size())
                    if not self.acmodel.always_broadcast:
                        self.rollout_broadcast_log_probs[j][i] = agents_broadcast_dist[j].logits[
                            range(self.num_procs), self.current_options[j].long(), agents_action[j].long(),
                            agents_broadcast[j].long()]

                    #self.rollout_masked_embeddings[j][i] = agents_broadcast_embedding[j]


                    if self.acmodel.recurrent:
                        self.rollout_agent_memories[j][i] = self.current_agent_memories[j]
                        self.current_agent_memories[j] = agents_memory[j]
                        # if not self.acmodel.use_teamgrid and not self.acmodel.use_central_critic:
                        #     self.rollout_particle_agent_memories[j][i] = self.particle_agent_memories[j]
                        #     self.particle_agent_memories[j] = particle_agents_memory[j]

                        if self.acmodel.use_central_critic:
                            self.rollout_coord_memories[i] = self.current_coord_memory
                            self.current_coord_memory = new_coord_memory



                    if self.acmodel.use_act_values:
                        if self.acmodel.always_broadcast:

                            self.rollout_values_swa[j][i] = agents_values[j][
                            range(self.num_procs), self.current_options[j].long(), agents_action[j].long()].squeeze()
                        else:
                            self.rollout_values_swa[j][i] = agents_values[j][
                                range(self.num_procs), self.current_options[j].long(), agents_action[j].long(),
                                agents_broadcast[j].long()].squeeze()

                        self.rollout_values_sw[j][i] = Qsw.squeeze()
                        self.rollout_values_s[j][i] = Vs.squeeze()
                        self.rollout_values_sw_max[j][i] = Qsw_max.squeeze()

                    else:
                        if self.acmodel.always_broadcast:
                            self.rollout_values[j][i] = agents_values[j][range(self.num_procs), self.current_options[j].long(), agents_action[j].long()].squeeze()
                        else:
                            self.rollout_values[j][i] = agents_values[j][
                                range(self.num_procs), self.current_options[j].long(), agents_action[
                                    j].long(), agents_broadcast[j].long()].squeeze()

                        # self.rollout_values_b[j][i] = agents_values_b[j][
                        #     range(self.num_procs), self.current_options[j].long()].squeeze()

                    if self.acmodel.use_term_fn:

                        self.rollout_terminates_prob[j][i] = agents_term_dist[j].probs[range(self.num_procs), self.current_options[j].long()]
                        self.rollout_terminates[j][i] = terminate


                    #if self.acmodel.use_broadcasting:
                        # #self.rollout_broadcast_probs[j][i] = agents_broadcast_dist[j].probs[range(self.num_procs), self.current_options[j].long()]
                    self.rollout_broadcast_masks[j][i] = agents_broadcast[j]






                        # self.rollout_values_swa_b[j][i] = agents_values_b[j][
                        #     range(self.num_procs), self.current_options[j].long(), agents_broadcast[j]].squeeze()
                        # self.rollout_values_sw_b[j][i] = Qsw_b.squeeze()
                        # self.rollout_values_s_b[j][i] = Vs_b.squeeze()
                        # self.rollout_values_sw_b_max[j][i] = Qsw_b_max.squeeze()


                    # if self.acmodel.use_central_critic:
                    #
                    #     self.rollout_coord_memories[i] = self.current_coord_memory
                    #     # x = new_coord_memories[range(self.num_procs), self.current_options[j].long(), agents_action[j].long(), :]
                    #     x = coord_new_memory
                    #     self.current_coord_memory = x

                #print('agents_action', agents_action, 'star_action', *agents_action_mlp)

                # environment step
                if self.acmodel.use_teamgrid:
                    next_obss, rewards, done, _ = self.env.step(list(map(list, zip(*agents_action))))  # this list(map(list)) thing is used to transpose a list of lists
                else:
                    next_obss, rewards, done, _ = self.env.step(list(map(list, zip(*agents_action_mlp))))

                #self.env_step += 1
                #print('i', i, 'self.env_step', self.env_step)
                #print('num_frames',self.acmodel.frames_per_proc)
                terminal = [(i >= self.acmodel.frames_per_proc) for _ in range(self.shape[1])]



                if not self.acmodel.use_teamgrid:
                    done = [all(item) for item in done]
                #print('done', done, 'terminal', terminal)
                # done = not done or terminal
                #done = [item1 or item2 for (item1, item2) in zip(done, terminal)]
               # print('done1', done, )
                    #print('i',i,'done_t', done)

                self.rollout_obss[i] = self.current_obss
                self.rollout_next_obss[i] = next_obss
                self.current_obss = next_obss


                self.rollout_masks[i] = self.current_mask
                done_or_term = [item1 or item2 for (item1, item2) in zip(done, terminal)]
                self.current_mask = 1. - torch.tensor(done_or_term, device=self.device, dtype=torch.float)
                #print('done', done, 'terminal', terminal, 'done or term', done_or_term, 'self.current_mask', self.current_mask)

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
                        #print('self.rollout_rewards[j][i]', self.rollout_rewards[j][i])
                        a = torch.tensor(reward, device=self.device)

                        b = torch.tensor(agents_broadcast[j].unsqueeze(1).float()*self.broadcast_penalty, device=self.device)

                        if self.acmodel.use_teamgrid:
                            self.rollout_rewards_plus_broadcast_penalties[j][i] = torch.add(a,b.squeeze().long())
                        else:
                            self.rollout_rewards_plus_broadcast_penalties[j][i] = torch.add(a, b.squeeze())


                    if self.acmodel.use_term_fn:
                        # change current_option w.r.t. episode ending

                        self.current_options[j] = self.current_mask * self.current_options[j] + \
                                                  (1. - self.current_mask) * torch.randint(low=0, high=self.num_options,
                                                                                         size=(self.num_procs,),
                                                                                         device=self.device,
                                                                                         dtype=torch.float)

                    # update log values

                    self.log_episode_return[j] += self.rollout_rewards[j][i] #torch.tensor(reward, device=self.device, dtype=torch.float)
                    #print('self.log_episode_return', self.log_episode_return)
                    self.log_episode_return_with_broadcast_penalties[j] += self.rollout_rewards_plus_broadcast_penalties[j][i]
                    self.log_episode_reshaped_return[j] += self.rollout_rewards[j][i]
                    self.log_episode_num_frames[j] += torch.ones(self.num_procs, device=self.device)

                    for k, done_ in enumerate(done):
                        if done_ or terminal:
                            #print('num_step', self.env_step)
                            #self.env_step = 0
                            self.log_done_counter[j] += 1
                            self.log_return[j].append(self.log_episode_return[j][k].item())
                            self.log_return_with_broadcast_penalties[j].append(self.log_episode_return_with_broadcast_penalties[j][k].item())
                            # self.log_episode_return_with_broadcast_penalties = [
                            #     torch.zeros(self.shape[1], device=self.device) for _ in range(self.num_agents)]
                            #print('log_return', self.log_return_with_broadcast_penalties[j])
                            self.log_reshaped_return[j].append(self.log_episode_reshaped_return[j][k].item())
                            self.log_num_frames[j].append(self.log_episode_num_frames[j][k].item())

                    #print('self.log_return',  self.log_return, 'self.current_mask', self.current_mask)

                    self.log_episode_return[j] *= self.current_mask
                    self.log_episode_return_with_broadcast_penalties[j] *= self.current_mask
                    #print('current_mask', self.current_mask, 'log_episode_return', self.log_episode_return_with_broadcast_penalties)
                    #print('self.log_episode_return_with_broadcast_penalties', self.log_episode_return_with_broadcast_penalties)
                    self.log_episode_reshaped_return[j] *= self.current_mask
                    #print('log_return', self.log_episode_return_with_broadcast_penalties[j])
                    self.log_episode_num_frames[j] *= self.current_mask

            # Add advantage and return to experiences

            for j in range(self.num_agents):

                for i in reversed(range(self.num_frames_per_proc)):
                    if self.acmodel.use_term_fn and self.acmodel.use_act_values:
                        #next_mask = self.rollout_masks[i + 1]

                        # # For coordinator  q-learning (centralized)
                        # agents_no_term_prob_array = np.array([1. - self.rollout_terminates_prob[j][i + 1].numpy() for j in range(self.num_agents)])
                        # no_agent_term_prob = agents_no_term_prob_array.prod()
                        #
                        #
                        #
                        # # the env-reward is copied for each agent, so we can take for any j self.rollout_rewards[j][i]
                        # interim = torch.stack(agents_broadcast)
                        # total_broadcast_list = np.sum(np.array(interim.tolist()), 0)
                        #
                        # self.rollout_coord_target[i] = self.rollout_rewards[0][i] - torch.tensor(total_broadcast_list*self.broadcast_penalty, device=self.device).float() + \
                        #                  self.rollout_masks[i + 1] * self.discount * \
                        #                  (
                        #                          no_agent_term_prob * self.rollout_coord_value_sw[i+1] + \
                        #                          (1. - no_agent_term_prob) * self.rollout_coord_value_sw_max[i+1]
                        #                  )
                        #
                        # self.rollout_coord_advantages[i] = self.rollout_coord_value_sw[i + 1] - self.rollout_coord_value_s[
                        #     i + 1]


                        # Centralized q learning for DOC
                        self.rollout_targets[j][i] = self.rollout_rewards_plus_broadcast_penalties[j][i] + \
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
                    elif not self.acmodel.use_term_fn and self.acmodel.use_act_values:
                        # Centralized q learning for COMA
                        self.rollout_targets[j][i] = self.rollout_rewards_plus_broadcast_penalties[j][i] + \
                                                     self.rollout_masks[i + 1] * self.discount * self.rollout_values_sw[j][i + 1]

                        # action-advantage for action

                        self.rollout_advantages[j][i] = self.rollout_values_swa[j][i] - self.rollout_values_sw[j][
                            i]

                    elif not self.acmodel.use_term_fn and not self.acmodel.use_act_values:
                        next_mask = self.rollout_masks[i+1]
                        next_value = self.rollout_values[j][i+1]

                        next_advantage = self.rollout_advantages[j][i+1] if i < self.num_frames_per_proc else 0


                        delta = self.rollout_rewards_plus_broadcast_penalties[j][i] + self.discount * next_value * self.rollout_masks[i + 1] - self.rollout_values[j][i]
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
            # dic = defaultdict(list)
            # exps = [dic for _ in range(self.num_agents)]

            if self.acmodel.use_central_critic:
                coord_exps = DictList()

            logs = {k:[] for k in ["return_per_episode",
                                   "return_per_episode_with_broadcast_penalties",
                                   "mean_agent_return_with_broadcast_penalties",
                                   "reshaped_return_per_episode",
                                   "num_frames_per_episode",
                                   "num_frames",
                                   "entropy",
                                   "broadcast_entropy",
                                   "broadcast_loss",
                                   "value",
                                   "policy_loss",
                                   "value_loss",
                                   "grad_norm",
                                   "options",
                                   "actions"]}


           #  import pdb;
           #  pdb.set_trace()
            # Get all agents successfully broadcast embeddings
            all_agents_successfully_broadcast_embedding = []
            for j in range(self.num_agents):
                exps[j].obs = [self.rollout_obss[i][j][k] for k in range(self.num_procs) \
                                               for i in range(self.num_frames_per_proc)]
                exps[j].next_obs = [self.rollout_next_obss[i][j][k] for k in range(self.num_procs) \
                                               for i in range(self.num_frames_per_proc)]

                # exps[j].obs = torch.tensor(self.rollout_obss).transpose(0, 1)[j][:-1].transpose(0, 1)
                # exps[j].next_obs = torch.tensor(self.rollout_next_obss).transpose(0, 1)[j][:-1].transpose(0, 1)

                exps[j].obs = self.preprocess_obss(exps[j].obs, device=self.device)
                exps[j].next_obs = self.preprocess_obss(exps[j].next_obs, device=self.device)


                if self.acmodel.recurrent:
                    # T x P x D -> P x T x D -> (P * T) x D
                    if self.acmodel.use_central_critic:
                        coord_exps.memory = self.rollout_coord_memories[:-1].transpose(0, 1).reshape(-1, *self.rollout_coord_memories.shape[2:])

                    exps[j].memory = self.rollout_agent_memories[j][:-1].transpose(0, 1).reshape(-1, *self.rollout_agent_memories[j].shape[2:])
                    # if not self.acmodel.use_teamgrid:
                    #     exps[j].particle_agent_memory = self.rollout_particle_agent_memories[j][:-1].transpose(0, 1).reshape(-1, *self.rollout_particle_agent_memories[j].shape[2:])
                    # T x P -> P x T -> (P * T) x 1
                    exps[j].mask = self.rollout_masks[:-1].transpose(0, 1).reshape(-1).unsqueeze(1)

                    # for all tensors below, T x P -> P x T -> P * T
                    exps[j].embedding = self.rollout_agent_embeddings[j][:-1].transpose(0, 1).reshape(-1, *self.rollout_agent_embeddings[j].shape[2:])
                    # if not self.acmodel.use_teamgrid:
                    #     exps[j].particle_agent_embedding = self.rollout_particle_agent_embeddings[j][:-1].transpose(0, 1).reshape(-1, *self.rollout_particle_agent_embeddings[j].shape[2:])
                    exps[j].masked_embedding = self.rollout_masked_embeddings[j][:-1].transpose(0, 1).reshape(-1, *
                    self.rollout_masked_embeddings[j].shape[2:])
                    exps[j].estimated_embedding = self.rollout_estimated_embeddings[j][:-1].transpose(0, 1).reshape(-1, *self.rollout_estimated_embeddings[j].shape[2:])

                    exps[j].action = self.rollout_actions[j][:-1].transpose(0, 1).reshape(-1) #.long()
                    exps[j].action_mlp = self.rollout_actions_mlp[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].option = self.rollout_options[j][:-1].transpose(0, 1).reshape(-1)
                    #exps[j].others_action = [self.rollout_actions[k][:-1].transpose(0, 1).reshape(-1) for k in range(self.num_agents) if k != j]
                    # if self.acmodel.use_broadcasting:
                    exps[j].broadcast = self.rollout_broadcast_masks[j][:-1].transpose(0, 1).reshape(-1) #.long()

                    exps[j].reward = self.rollout_rewards[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].reward_plus_broadcast_penalties = self.rollout_rewards_plus_broadcast_penalties[j][:-1].transpose(0, 1).reshape(-1) # this is what updates critic
                    # exps[j].all_reward_plus_broadcast_penalties = self.rollout_rewards_plus_broadcast_penalties[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].advantage = self.rollout_advantages[j][:-1].transpose(0, 1).reshape(-1)
                    # exps[j].advantage_b = self.rollout_advantages_b[j][:-1].transpose(0, 1).reshape(-1)

                    exps[j].log_prob = self.rollout_log_probs[j][:-1].transpose(0, 1).reshape(-1)
                    exps[j].broadcast_log_prob = self.rollout_broadcast_log_probs[j][:-1].transpose(0, 1).reshape(
                        -1)
                    exps[j].current_options = self.rollout_options[j][:-1].transpose(0, 1).reshape(-1).long()

                    if self.acmodel.use_term_fn:
                        exps[j].terminate = self.rollout_terminates[j][:-1].transpose(0, 1).reshape(-1)
                        exps[j].terminate_prob = self.rollout_terminates_prob[j][:-1].transpose(0, 1).reshape(-1)

                    if self.acmodel.use_act_values:

                        # coord_exps.value_swa = self.rollout_coord_value_swa[:-1].transpose(0, 1).reshape(-1)
                        # coord_exps.value_sw = self.rollout_coord_value_sw[:-1].transpose(0, 1).reshape(-1)
                        # coord_exps.value_s = self.rollout_coord_value_s[:-1].transpose(0, 1).reshape(-1)
                        # coord_exps.advantage = self.rollout_coord_advantages[:-1].transpose(0, 1).reshape(-1)

                        exps[j].value_swa = self.rollout_values_swa[j][:-1].transpose(0, 1).reshape(-1)
                        exps[j].value_sw = self.rollout_values_sw[j][:-1].transpose(0, 1).reshape(-1)
                        exps[j].value_s = self.rollout_values_s[j][:-1].transpose(0, 1).reshape(-1)

                        # exps[j].value_swa_b = self.rollout_values_swa_b[j][:-1].transpose(0, 1).reshape(-1)
                        # exps[j].value_sw_b = self.rollout_values_sw_b[j][:-1].transpose(0, 1).reshape(-1)

                        exps[j].target = self.rollout_targets[j][:-1].transpose(0, 1).reshape(-1)
                        # exps[j].target_b = self.rollout_targets_b[j][:-1].transpose(0, 1).reshape(-1)

                        # coord_exps.target = self.rollout_coord_target[:-1].transpose(0, 1).reshape(-1)


                    else:
                        exps[j].value = self.rollout_values[j][:-1].transpose(0, 1).reshape(-1)
                        # exps[j].value_b = self.rollout_values_b[j][:-1].transpose(0, 1).reshape(-1)
                        exps[j].returnn = exps[j].value + exps[j].advantage

                    # # Preprocess experiences
                    #
                    # exps[j].obs = self.preprocess_obss(exps[j].obs, device=self.device)
                    # exps[j].next_obs = self.preprocess_obss(exps[j].obs, device=self.device)


                # else:
                #     if self.acmodel.recurrent:
                #         # T x P x D
                #         if self.acmodel.use_central_critic:
                #             coord_exps.memory = self.rollout_coord_memories[:-1].transpose(0, 1).reshape(-1,
                #                                                                                          *self.rollout_coord_memories.shape[
                #                                                                                           2:])
                #
                #         exps[j].memory = self.rollout_agent_memories[j][:-1].transpose(0, 1).reshape(-1, *
                #         self.rollout_agent_memories[j].shape[2:])
                #
                #         # T x P -> P x T -> (P * T) x 1
                #         exps[j].mask = self.rollout_masks[:-1] #.transpose(0, 1).reshape(-1).unsqueeze(1)
                #
                #         # for all tensors below, T x P
                #         exps[j].embedding = self.rollout_agent_embeddings[j][:-1] #.transpose(0, 1).reshape(-1, *
                #         #self.rollout_agent_embeddings[j].shape[2:])
                #         exps[j].masked_embedding = self.rollout_masked_embeddings[j][:-1] #.transpose(0, 1).reshape(-1, *
                #         #self.rollout_masked_embeddings[j].shape[2:])
                #         exps[j].estimated_embedding = self.rollout_estimated_embeddings[j][:-1] #.transpose(0, 1).reshape(
                #            # -1, *self.rollout_estimated_embeddings[j].shape[2:])
                #         exps[j].action = self.rollout_actions[j][:-1] #.transpose(0, 1).reshape(-1)  # .long()
                #         # if self.acmodel.use_broadcasting:
                #         exps[j].broadcast = self.rollout_broadcast_masks[j][:-1] #.transpose(0, 1).reshape(-1)  # .long()
                #
                #         exps[j].reward = self.rollout_rewards[j][:-1] #.transpose(0, 1).reshape(-1)
                #         exps[j].reward_plus_broadcast_penalties = self.rollout_rewards_plus_broadcast_penalties[j] #[
                #                                                   #:-1].transpose(0, 1).reshape(
                #            # -1)  # this is what updates critic
                #         # exps[j].all_reward_plus_broadcast_penalties = self.rollout_rewards_plus_broadcast_penalties[j][:-1].transpose(0, 1).reshape(-1)
                #         exps[j].advantage = self.rollout_advantages[j][:-1] #.transpose(0, 1).reshape(-1)
                #         # exps[j].advantage_b = self.rollout_advantages_b[j][:-1].transpose(0, 1).reshape(-1)
                #
                #         exps[j].log_prob = self.rollout_log_probs[j][:-1] #.transpose(0, 1).reshape(-1)
                #         exps[j].broadcast_log_prob = self.rollout_broadcast_log_probs[j][:-1] #.transpose(0, 1).reshape(
                #             #-1)
                #         exps[j].current_options = self.rollout_options[j][:-1] #.transpose(0, 1).reshape(-1).long()
                #
                #         if self.acmodel.use_term_fn:
                #             exps[j].terminate = self.rollout_terminates[j][:-1] #.transpose(0, 1).reshape(-1)
                #             exps[j].terminate_prob = self.rollout_terminates_prob[j][:-1] #.transpose(0, 1).reshape(-1)
                #
                #         if self.acmodel.use_act_values:
                #
                #             # coord_exps.value_swa = self.rollout_coord_value_swa[:-1].transpose(0, 1).reshape(-1)
                #             # coord_exps.value_sw = self.rollout_coord_value_sw[:-1].transpose(0, 1).reshape(-1)
                #             # coord_exps.value_s = self.rollout_coord_value_s[:-1].transpose(0, 1).reshape(-1)
                #             # coord_exps.advantage = self.rollout_coord_advantages[:-1].transpose(0, 1).reshape(-1)
                #
                #             exps[j].value_swa = self.rollout_values_swa[j][:-1] #.transpose(0, 1).reshape(-1)
                #             exps[j].value_sw = self.rollout_values_sw[j][:-1] #.transpose(0, 1).reshape(-1)
                #             exps[j].value_s = self.rollout_values_s[j][:-1] #.transpose(0, 1).reshape(-1)
                #
                #             # exps[j].value_swa_b = self.rollout_values_swa_b[j][:-1].transpose(0, 1).reshape(-1)
                #             # exps[j].value_sw_b = self.rollout_values_sw_b[j][:-1].transpose(0, 1).reshape(-1)
                #
                #             exps[j].target = self.rollout_targets[j][:-1] #.transpose(0, 1).reshape(-1)
                #             # exps[j].target_b = self.rollout_targets_b[j][:-1].transpose(0, 1).reshape(-1)
                #
                #             # coord_exps.target = self.rollout_coord_target[:-1].transpose(0, 1).reshape(-1)
                #
                #
                #         else:
                #             exps[j].value = self.rollout_values[j][:-1] #.transpose(0, 1).reshape(-1)
                #             # exps[j].value_b = self.rollout_values_b[j][:-1].transpose(0, 1).reshape(-1)
                #             exps[j].returnn = exps[j].value + exps[j].advantage
                #
                #         # # Preprocess experiences
                #         #
                #         # exps[j].obs = self.preprocess_obss(exps[j].obs, device=self.device)
                #         # exps[j].next_obs = self.preprocess_obss(exps[j].obs, device=self.device)

                # Log some values

                keep = max(self.log_done_counter[j], self.num_procs)
                #print('self.log_return', self.log_return_with_broadcast_penalties, 'keep', keep)

                logs["return_per_episode"].append(self.log_return[j][-keep:])
                #print('logs["return_per_episode"]', logs["return_per_episode"])
                logs["return_per_episode_with_broadcast_penalties"].append(self.log_return_with_broadcast_penalties[j][-keep:]) # this is what we plot
                #print('keep', keep, 'base_log_return', logs["return_per_episode_with_broadcast_penalties"])

                #print('shape', np.shape(self.log_return_with_broadcast_penalties[j][-keep:]))
                logs["reshaped_return_per_episode"].append(self.log_reshaped_return[j][-keep:])
                logs["num_frames_per_episode"].append(self.log_num_frames[j][-keep:])
                logs["num_frames"].append(self.num_frames)

                self.log_return[j] = self.log_return[j][-self.num_procs:]
                self.log_return_with_broadcast_penalties[j] = self.log_return_with_broadcast_penalties[j][-self.num_procs:]
                self.log_reshaped_return[j] = self.log_reshaped_return[j][-self.num_procs:]
                self.log_num_frames[j] = self.log_num_frames[j][-self.num_procs:]
            self.log_mean_agent_return = list(np.mean(self.log_return_with_broadcast_penalties, axis=0))[-keep:]
            #print('self.log_mean_agent_return_shape', np.shape(self.log_return_with_broadcast_penalties), 'shape_mean_return', np.shape(self.log_mean_agent_return))
            logs["mean_agent_return_with_broadcast_penalties"].append(self.log_mean_agent_return)
        if self.acmodel.use_central_critic:
            return coord_exps, exps, logs
        else:
            return exps, logs
        #return coord_exps, exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
