import numpy as np
from torch import Tensor
from torch.autograd import Variable
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import random
import torch


class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    op_dims = 0 if not DOC or OC
    """
    def __init__(self, env_id, max_steps, num_agents, obs_dims, op_dims, ac_dims, br_dims, ep_buffer):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            op_dims (list of ints): number of option dimensions for each
                                     agent. None if not DOC or OC
            ac_dims (list of ints): number of action dimensions for each agent
            br_dims (list of ints): number of broadcast dimensions for each agent

        """
        self.env_id = env_id
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.ep_buffer = ep_buffer
        self.obs_buffs = []
        self.old_embeddings_buffs = []
        if op_dims is not None:
            self.ops_buffs = []
        self.beta_buffs = []
        self.ac_buffs = []
        self.br_buffs = []
        self.mem_buffs = []
        self.rew_buffs = []
        self.br_rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []

        if self.env_id == 'cleanup' or self.env_id == 'harvest':
            self.obs_dim_embed = ((obs_dims[0][0]-1)//2 -3)*((obs_dims[0][1]-1)//2 -3)*32
        elif self.env_id == 'simple_spread':
            self.obs_dim_embed = obs_dims[0]
        elif self.env_id == 'simple_speaker_listener':
            self.obs_dim_embed = obs_dims
        else: #teamgrid
            self.obs_dim_embed = ((obs_dims[0][0]-1)//2 -2)*((obs_dims[0][1]-1)//2 -2)*64

        tot_dim=0
        if env_id=='simple_speaker_listener':
            for i in range(len(obs_dims)):
                tot_dim += 2 * (self.obs_dim_embed[i] + ac_dims[i] + br_dims[i])
            self.cr_mem_buff = np.zeros((max_steps, tot_dim))
        elif env_id=='cleanup' or env_id=='harvest' or env_id=='simple_spread':
            self.cr_mem_buff = np.zeros((max_steps, 2 * len(obs_dims) * (self.obs_dim_embed + ac_dims[0] + br_dims[0])))
        else: # teamgrid
            self.cr_mem_buff = np.zeros((max_steps, 2 * len(obs_dims) * (self.obs_dim_embed + op_dims[0] + ac_dims[0] \
                                                                         + br_dims[0])))
            #all agents have same action and broadcast dimessions
        for i, odim, opdim, adim, bdim in zip(range(len(obs_dims)), obs_dims, op_dims, ac_dims, br_dims):
            if self.env_id == 'cleanup' or self.env_id == 'harvest':
                self.obs_buffs.append(np.zeros((max_steps, odim[0], odim[1], odim[2])))
                self.old_embeddings_buffs.append(np.zeros((max_steps, self.obs_dim_embed)))
            elif self.env_id == 'simple_spread':
                self.obs_buffs.append(np.zeros((max_steps, odim)))
                self.old_embeddings_buffs.append(np.zeros((max_steps, self.obs_dim_embed)))
            elif self.env_id == 'simple_speaker_listener':
                self.obs_buffs.append(np.zeros((max_steps, odim)))
                self.old_embeddings_buffs.append(np.zeros((max_steps, self.obs_dim_embed[i])))
            else: #teamgrid
                self.obs_buffs.append(np.zeros((max_steps, odim[0], odim[1], odim[2])))
                self.old_embeddings_buffs.append(np.zeros((max_steps, self.obs_dim_embed)))
            if op_dims is not None:
                self.ops_buffs.append(np.zeros((max_steps, opdim)))
            self.beta_buffs.append(np.zeros((max_steps, opdim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.br_buffs.append(np.zeros((max_steps, bdim)))
            if self.env_id == 'cleanup' or self.env_id == 'harvest':
                memdim = ((odim[0]-1)//2 -3)*((odim[1]-1)//2 -3)*32 * 2
            elif self.env_id=='simple_spread' or self.env_id=='simple_speaker_listener':
                memdim = odim * 2
            else: # teamgrid
                memdim = ((odim[0]-1)//2 -2)*((odim[1]-1)//2 -2)*64 * 2
            self.mem_buffs.append(np.zeros((max_steps, memdim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.br_rew_buffs.append(np.zeros(max_steps))

            if self.env_id == 'cleanup' or self.env_id == 'harvest':
                self.next_obs_buffs.append(np.zeros((max_steps, odim[0], odim[1], odim[2])))
            elif self.env_id == 'simple_spread' or self.env_id=='simple_speaker_listener':
                self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            else:  # teamgrid
                self.next_obs_buffs.append(np.zeros((max_steps, odim[0], odim[1], odim[2])))
            self.done_buffs.append(np.zeros(max_steps))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    '''options = None if not DOC or OC'''
    def push(self, observations, old_embeddings, options, actions, broadcasts, betas, agents_memories, cr_memory, rewards, \
             br_rewards, next_observations, dones): # input is one-hot_options; options = None if not doc or oc

        if not self.ep_buffer:
            nentries = len(observations[0]) # handle multiple parallel environments
        else:
            nentries = 1

        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.cr_mem_buff = np.roll(self.cr_mem_buff, rollover)
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.old_embeddings_buffs[agent_i] = np.roll(self.old_embeddings_buffs[agent_i],
                                                  rollover, axis=0)
                if options is not None:
                    self.ops_buffs[agent_i] = np.roll(self.ops_buffs[agent_i],
                                                  rollover, axis=0)
                self.beta_buffs[agent_i] = np.roll(self.beta_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.br_buffs[agent_i] = np.roll(self.br_buffs[agent_i],
                                                 rollover, axis=0)
                self.mem_buffs[agent_i] = np.roll(self.mem_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.br_rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
                #self.done_buffs[self.num_agents] = dones[0]['__all__']

            self.curr_i = 0
            self.filled_i = self.max_steps

        if not self.ep_buffer:
            self.cr_mem_buff[self.curr_i:self.curr_i + nentries] = cr_memory.detach()
            for agent_i in range(self.num_agents):
                #import pdb; pdb.set_trace()
                self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observations[agent_i]
                if old_embeddings[0][0] is not None: # If the fist element is None, then so it is for the rest of the elements
                    self.old_embeddings_buffs[agent_i][self.curr_i:self.curr_i + nentries] = old_embeddings[agent_i].detach()

                # options, actions and broadcasts are already batched by agent, so they are indexed differently
                if options is not None:
                    self.ops_buffs[agent_i][self.curr_i:self.curr_i + nentries] = options[agent_i]
                self.beta_buffs[agent_i][self.curr_i:self.curr_i + nentries] = betas[agent_i].detach()
                self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
                self.br_buffs[agent_i][self.curr_i:self.curr_i + nentries] = broadcasts[agent_i]

                self.mem_buffs[agent_i][self.curr_i:self.curr_i + nentries] = agents_memories[agent_i].detach()

                self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]

                self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observations[agent_i]

                if self.env_id == 'cleanup' or self.env_id == 'harvest':
                    self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[0]['agent-'+str(agent_i)]
                else: # particle
                    self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[agent_i]

        else:
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observations[agent_i]
                if not all(a is None for a in old_embeddings):
                    self.old_embeddings_buffs[agent_i][self.curr_i:self.curr_i + nentries] = old_embeddings[
                        agent_i].detach()
                else:
                    self.old_embeddings_buffs[agent_i][self.curr_i:self.curr_i + nentries] = old_embeddings[agent_i]

                # options, actions and broadcasts are already batched by agent, so they are indexed differently
                if options is not None:
                    self.ops_buffs[agent_i][self.curr_i:self.curr_i + nentries] = options[agent_i]
                self.beta_buffs[agent_i][self.curr_i:self.curr_i + nentries] = betas[agent_i].detach()
                self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
                self.br_buffs[agent_i][self.curr_i:self.curr_i + nentries] = broadcasts[agent_i]

                self.mem_buffs[agent_i][self.curr_i:self.curr_i + nentries] = agents_memories[agent_i].detach()

                self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
                self.br_rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = br_rewards[agent_i]
                self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observations[agent_i]

                self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[agent_i]


        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, batch_size):
        inds = np.random.choice(np.arange(self.filled_i), size=batch_size, replace=False)
        return self._encode_sample(inds)

    def _encode_sample(self, inds, to_gpu=False, norm_rews=True):
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
            ret_br_rews = [cast((self.br_rew_buffs[i][inds] -
                              self.br_rew_buffs[i][:self.filled_i].mean()) /
                             self.br_rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
            br_ret_rews = [cast(self.br_rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.old_embeddings_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ops_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.br_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.beta_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.mem_buffs[i][inds]) for i in range(self.num_agents)],
                cast(self.cr_mem_buff[inds]),
                ret_rews,
                br_ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, batch_size):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - batch_size, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - batch_size), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, env_id, max_steps, num_agents, obs_dims, op_dims, ac_dims, br_dims,  alpha, ep_buffer):
        super(PrioritizedReplayBuffer, self).__init__(env_id, max_steps, num_agents, obs_dims, op_dims, ac_dims, br_dims, ep_buffer)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < max_steps:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args, **kwargs):
        idx = self.curr_i
        super().push(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, self.max_steps - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):

            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < self.max_steps
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, num_agents, beta):

        idxes = self._sample_proportional(batch_size)

        if beta > 0:
            weights = [[] for _ in range(num_agents)]
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * self.max_steps) ** (-beta)

            for idx in idxes:
                for i in range(num_agents):
                    p_sample = self._it_sum[idx] / self._it_sum.sum()
                    weight = (p_sample * self.max_steps) ** (-beta)
                    weights[i].append((weight / max_weight))
            weights = np.array(weights)
        else:
            weights = np.array([[1. for _ in range(num_agents)] for _ in range(len(idxes))], dtype=np.float32)
        encoded_sample = self._encode_sample(idxes, to_gpu=False, norm_rews=False) # TODO:make norm_rews=True for experimenst
        return tuple(list(encoded_sample) + [weights, idxes])


