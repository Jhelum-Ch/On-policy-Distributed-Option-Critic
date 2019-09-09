import numpy as np
import torch
import torch.nn.functional as F

from torch_rl.algos.base import BaseAlgo
from torch_rl.algos.replayBuffer import ReplayBuffer

class MADDPGAlgo(BaseAlgo):
    """The class for the Multi-Agent Deep Deterministic Policy Gradient algorithm
       (cite)."""

    def __init__(self, num_agents=None, envs=None, acmodel=None, replay_buffer=None, num_frames_per_proc=None, discount=0.99, lr=7e-4,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, er_batch_size=256, preprocess_obss=None, num_options=3,
                 termination_reg=0.01, termination_loss_coef=0.5, reshape_reward=None, always_broadcast=False, broadcast_penalty=-0.01):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(num_agents=num_agents, envs=envs, acmodel=acmodel, replay_buffer=replay_buffer, \
                         num_frames_per_proc=num_frames_per_proc, discount=discount, lr=lr, gae_lambda=gae_lambda,
                         entropy_coef=entropy_coef,
                         value_loss_coef=value_loss_coef, max_grad_norm=max_grad_norm, recurrence=recurrence, \
                         preprocess_obss=preprocess_obss, reshape_reward=reshape_reward, broadcast_penalty=broadcast_penalty,
                         termination_reg=termination_reg, termination_loss_coef=termination_loss_coef)

        # if not self.acmodel.use_teamgrid and not self.acmodel.use_central_critic:
        #     a = self.acmodel.parametersList
        #     self.optimizer = torch.optim.RMSprop(a, lr, alpha=rmsprop_alpha, eps=rmsprop_alpha)
        # else:
        #     self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
        #                                          alpha=rmsprop_alpha, eps=rmsprop_eps)

        if not self.acmodel.use_teamgrid and not self.acmodel.use_central_critic:
            a = self.acmodel.parametersList
            self.optimizer = torch.optim.Adam(a, lr, eps=adam_eps)
        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)

        self.num_agents = num_agents
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.er_batch_size = er_batch_size # how many samples from replay buffer
        #assert self.batch_size % self.recurrence == 0


        # make experience buffer
        self.replayBuffer = ReplayBuffer(1e6)

        self.max_replayBuffer_len = self.er_batch_size * num_frames_per_proc
        self.replay_sample_index = None

        self.batch_num = 0

    def update_parameters(self):

        # Collect experiences
        if self.acmodel.use_central_critic:
            coord_exps, exps, logs = self.collect_experiences()

        else:
            exps, logs = self.collect_experiences()


        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize containers for actors info

        sbs_coord = None
        sbs = [None for _ in range(self.num_agents)]
        embeddings = [None for _ in range(self.num_agents)]
        next_embeddings = [None for _ in range(self.num_agents)]
        masked_embeddings = [None for _ in range(self.num_agents)]
        estimated_embeddings = [None for _ in range(self.num_agents)]
        next_estimated_embeddings = [None for _ in range(self.num_agents)]


        # All entities required for experience replay
        if self.acmodel.use_teamgrid:
            all_embeddings_er = [torch.zeros(self.er_batch_size, self.acmodel.semi_memory_size, device=self.device) \
                                 for _ in range(self.num_agents)]
            all_next_embeddings_er = [torch.zeros(self.er_batch_size, self.acmodel.semi_memory_size, device=self.device) \
                                 for _ in range(self.num_agents)]
            #print('all_embeddings_er_size', np.shape(all_embeddings_er[0][]))
            all_est_embeddings_er = [torch.zeros(self.er_batch_size, self.acmodel.semi_memory_size, device=self.device) \
                                 for _ in range(self.num_agents)]
            all_actions_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_next_actions_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_broadcasts_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_next_broadcasts_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_rewards_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_agents_values_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_next_agents_values_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)

            option_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                                 for _ in range(self.num_agents)]
            action_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                                 for _ in range(self.num_agents)]
            next_action_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                           for _ in range(self.num_agents)]
            broadcast_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                                 for _ in range(self.num_agents)]
            next_broadcast_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                              for _ in range(self.num_agents)]




            if self.acmodel.use_central_critic:
                sbs_coord = None
                update_critic_loss = 0
                update_value = 0
                update_value_loss = 0
                # option_idxs = [None for _ in range(self.num_agents)]
                # action_idxs = [None for _ in range(self.num_agents)]
                # broadcast_idxs = [None for _ in range(self.num_agents)]
            else:
                update_value = [0 for _ in range(self.num_agents)]
                update_value_loss = [0 for _ in range(self.num_agents)]
                update_critic_loss = [0 for _ in range(self.num_agents)]

            if self.acmodel.recurrent:
                memories = [exps[j].memory[inds] for j in range(self.num_agents)]

            update_entropy = [0 for _ in range(self.num_agents)]
            update_broadcast_entropy = [0 for _ in range(self.num_agents)]
            update_policy_loss = [0 for _ in range(self.num_agents)]
            update_broadcast_loss = [0 for _ in range(self.num_agents)]
            update_actor_loss = [0 for _ in range(self.num_agents)]

            for i in range(self.recurrence):
                if self.acmodel.use_central_critic:
                    sbs_coord = coord_exps[inds + i]


                for j in range(self.num_agents):
                    # Initialize memory

                    if self.acmodel.recurrent:
                        memory = memories[j]


                        sbs[j] = exps[j][inds + i]
                        #print('done', sbs[j].mask.size())

                        # Compute loss
                        # if len(self.replayBuffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
                        #     return
                        # if not t % 100 == 0:  # only update every 100 steps
                        #     return

                        self.replay_sample_index = self.replayBuffer.make_index(sbs[j], self.er_batch_size)


                        policy_loss = 0.0
                        value_loss = 0.0
                        loss = 0.0
                        entropy = 0.0
                        broadcast_entropy = 0.0


                        for ind, k in enumerate(self.replay_sample_index): # batch_size is the number os samples drawn from agent's buffer

                            agent_obs = sbs[j].obs[k]
                            agent_obs.image = agent_obs.image.unsqueeze(0)
                            agent_act = sbs[j].action[k].unsqueeze(0)
                            action_idxs[j][ind] = agent_act
                            agent_brd = sbs[j].broadcast[k].unsqueeze(0)
                            broadcast_idxs[j][ind] = agent_brd
                            agent_reward = sbs[j].reward_plus_broadcast_penalties[k].unsqueeze(0)
                            agent_next_obs = sbs[j].next_obs[k]
                            agent_next_obs.image = agent_next_obs.image.unsqueeze(0)
                            agent_advantage = sbs[j].advantage[k].unsqueeze(0)
                            agent_est_embedding = sbs[j].estimated_embedding[k].unsqueeze(0)
                            agent_done = sbs[j].mask[k].unsqueeze(0)
                            agent_memory = sbs[j].memory[k].unsqueeze(0)
                            option_in_use = sbs[j].current_options[k].unsqueeze(0)

                            option_idxs[j][ind] = option_in_use

                            if self.acmodel.recurrent:
                                if not self.acmodel.always_broadcast:
                                    act_dist, act_values, act_values_b, memory, _, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                                        agent_obs, agent_memory * (1. * agent_done), agent_index=j)
                                    next_act_dist, next_act_values, next_act_values_b, next_memory, _, next_broadcast_dist, next_embedding = self.acmodel.forward_agent_critic(
                                        agent_next_obs, agent_memory * (1. * agent_done), agent_index=j)
                                else:
                                    act_dist, act_values, memory, _, embedding = self.acmodel.forward_agent_critic(agent_obs,
                                                                                                                   agent_memory * (1. * agent_done),
                                                                                                                   agent_index=j)
                                    next_act_dist, next_act_values, next_memory, _, next_embedding = self.acmodel.forward_agent_critic(
                                        agent_next_obs,
                                        agent_memory * (1. * agent_done),
                                        agent_index=j)
                            else:
                                if not self.acmodel.always_broadcast:
                                    act_dist, act_values, act_values_b, _, _, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                                        agent_obs, agent_memory * (1. * agent_done), agent_index=j)
                                    next_act_dist, next_act_values, next_act_values_b, _, _, next_broadcast_dist, next_embedding = self.acmodel.forward_agent_critic(
                                        agent_next_obs, agent_memory * (1. * agent_done), agent_index=j)
                                else:
                                    act_dist, act_values, _, _, embedding = self.acmodel.forward_agent_critic(agent_obs,
                                                                                                              agent_index=j)
                                    next_act_dist, next_act_values, _, _, next_embedding = self.acmodel.forward_agent_critic(agent_next_obs,
                                                                                                              agent_index=j)

                            #print('option_in_use', torch.tensor(option_in_use.item()), 'option_idxs[j][ind].long()', option_idxs[j][ind].long())
                            next_agent_act = next_act_dist.sample()[option_in_use].squeeze(1)
                            #print('next_agent_act', next_agent_act.size(), 'next_action_idxs[j][ind]', next_action_idxs[j][ind])
                            next_action_idxs[j][ind] = next_agent_act
                            entropy += act_dist.entropy().mean()
                            agent_act_log_probs = \
                            act_dist.log_prob(agent_act.view(-1, 1).repeat(1, self.num_options))[
                                range(len(agent_act)), option_in_use]


                            policy_loss += -(agent_act_log_probs * agent_advantage).mean()


                            agent_values = act_values[range(len(agent_act)), option_in_use]
                            agent_next_values = next_act_values[range(len(agent_act)), option_in_use]

                            all_embeddings_er[j][ind] = embedding
                            all_next_embeddings_er[j][ind] = next_embedding
                            all_est_embeddings_er[j][ind] = agent_est_embedding
                            all_actions_er[j][ind] = agent_act
                            all_next_actions_er[j][ind] = next_agent_act
                            all_broadcasts_er[j][ind] = agent_brd
                            all_rewards_er[j][ind] = agent_reward
                            all_agents_values_er[j][ind] = agent_values
                            all_next_agents_values_er[j][ind] = agent_next_values

                            if not self.acmodel.use_central_critic:
                                delta = agent_reward + self.discount * agent_next_values - agent_values
                                # value_loss = (agent_values - sbs[j].returnn).pow(2).mean()
                                value_loss += delta.pow(2).mean()

                                update_value[j] += agent_values.mean().item()
                                update_value_loss[j] += value_loss.item()


                            if not self.acmodel.always_broadcast:
                                next_agent_brd = next_broadcast_dist.sample().squeeze()[next_agent_act]
                                next_broadcast_idxs[j][ind] = next_agent_brd
                                all_next_broadcasts_er[j][ind] = next_agent_brd

                                broadcast_entropy += broadcast_dist.entropy().mean()
                                broadcast_log_probs = broadcast_dist.log_prob(
                                    sbs[j].broadcast.view(-1, 1, 1).repeat(1, self.num_options, self.num_actions))[
                                    range(len(agent_brd)), option_in_use, agent_act.long()]
                                broadcast_loss = -(broadcast_log_probs * agent_advantage).mean()

                                loss += policy_loss - self.entropy_coef * entropy \
                                       + broadcast_loss \
                                       - self.entropy_coef * broadcast_entropy
                            else:
                                loss += policy_loss - self.entropy_coef * entropy



                        # sample averages
                        policy_loss /= self.er_batch_size
                        loss /= self.er_batch_size
                        entropy /= self.er_batch_size
                        broadcast_entropy /= self.er_batch_size

                        if not self.acmodel.use_central_critic:
                            value_loss /= self.er_batch_size
                            update_critic_loss[j] += self.value_loss_coef * value_loss
                            update_value[j] /= self.er_batch_size
                            update_value_loss[j] /= self.er_batch_size


                        # Update batch values

                        update_entropy[j] += entropy.item()
                        if not self.acmodel.always_broadcast:
                            update_broadcast_entropy[j] += broadcast_entropy.item()
                            update_broadcast_loss[j] += broadcast_loss.item()

                        update_policy_loss[j] += policy_loss.item()
                        update_actor_loss[j] += loss





                if self.acmodel.use_central_critic:
                    # Collect agent embeddings

                    for ind, k in enumerate(self.replay_sample_index):
                        op_inds = []
                        ac_inds = []
                        next_ac_inds = []
                        brd_inds = []
                        next_brd_inds = []

                        for j in range(self.num_agents):
                            op_inds.append(option_idxs[j][ind].unsqueeze(0))
                            ac_inds.append(action_idxs[j][ind].unsqueeze(0))
                            next_ac_inds.append(next_action_idxs[j][ind].unsqueeze(0))
                            brd_inds.append(broadcast_idxs[j][ind].unsqueeze(0))
                            next_brd_inds.append(next_broadcast_idxs[j][ind].unsqueeze(0))


                            embeddings[j] = all_embeddings_er[j][ind]
                            next_embeddings[j] = all_next_embeddings_er[j][ind]

                            # Collect masked (coord) embedding

                            if self.acmodel.always_broadcast:
                                masked_embeddings[j] = all_embeddings_er[j][ind].unsqueeze(0)
                                estimated_embeddings[j] = all_embeddings_er[j][ind].unsqueeze(0)
                                next_estimated_embeddings[j] = all_next_embeddings_er[j][ind].unsqueeze(0)
                            else:
                                masked_embeddings[j] = all_broadcasts_er[j][ind].unsqueeze(0) * all_embeddings_er[j][ind].unsqueeze(0)
                                # estimated_embeddings[j] = all_broadcasts_er[j][k].unsqueeze(1) * all_embeddings_er[j][k] + (1. - sbs[j].broadcast.unsqueeze(1)) * sbs[j].embedding
                                estimated_embeddings[j] = all_est_embeddings_er[j][ind].unsqueeze(0)

                                with torch.no_grad():
                                    next_estimated_embeddings[j] = (next_broadcast_idxs[j][ind] * next_embeddings[j]).unsqueeze(0) + \
                                                               ((1. - next_broadcast_idxs[j][ind]) * embeddings[j]).unsqueeze(0)


                        _, value_a_b, _ = self.acmodel.forward_central_critic(estimated_embeddings, op_inds,
                                                                              ac_inds, brd_inds, sbs_coord.memory[k].unsqueeze(0))
                        _, next_value_a_b, _ = self.acmodel.forward_central_critic(next_estimated_embeddings, op_inds,
                                                                              next_ac_inds, next_brd_inds,
                                                                              sbs_coord.memory[k].unsqueeze(0))

                        avg_value_loss = 0.0
                        for j in range(self.num_agents):
                            # avg_value_loss = avg_value_loss + (value_a_b - sbs[j].returnn).pow(2).mean()
                            # delta1 = all_rewards_er[j][ind] + self.discount * all_next_agents_values_er[j][ind] - all_agents_values_er[j][ind]
                            delta1 = all_rewards_er[j][ind] + self.discount * next_value_a_b - \
                                     value_a_b
                            avg_value_loss = avg_value_loss + delta1.pow(2).mean()
                        value_loss += avg_value_loss / self.num_agents

                        # update_value += coord_value_action.mean().item()
                        update_value += value_a_b.mean().item()

                    value_loss /= self.er_batch_size
                    update_critic_loss += self.value_loss_coef * value_loss
                    #print('update_critic_loss', update_critic_loss)
                    update_value /= self.er_batch_size
                    update_value_loss += value_loss.item()



            self.optimizer.zero_grad()

            # Actors back propagation

            for j in range(self.num_agents):

                update_entropy[j] /= self.recurrence  # recurrence_agents

                update_policy_loss[j] /= self.recurrence
                if not self.always_broadcast:
                    update_broadcast_entropy[j] /= self.recurrence
                    update_broadcast_loss[j] /= self.recurrence

                update_actor_loss[j] /= self.recurrence

                # update_values_b[j] /= self.recurrence

                update_actor_loss[j].backward(retain_graph=True)

            # Critic back propagation
            if self.acmodel.use_central_critic:
                update_value /= self.recurrence  # recurrence_coord
                update_value_loss /= self.recurrence
                update_critic_loss.backward()
            else:
                for j in range(self.num_agents):
                    update_value[j] /= self.recurrence  # recurrence_coord
                    update_value_loss[j] /= self.recurrence
                    update_critic_loss[j].backward(retain_graph=True)

            for name, param in self.acmodel.named_parameters():
                # print('name', name, 'param.data', param.data, 'param_grad', param.grad)
                if param.grad is None:
                    print('Grad_none', name)
            update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
            self.optimizer.step()


        else: # for particle env
            all_embeddings_er = [torch.zeros(self.er_batch_size, self.acmodel.semi_memory_size[j], device=self.device) \
                                 for j in range(self.num_agents)]
            all_next_embeddings_er = [torch.zeros(self.er_batch_size, self.acmodel.semi_memory_size[j], device=self.device) \
                                      for j in range(self.num_agents)]
            # print('all_embeddings_er_size', np.shape(all_embeddings_er[0][]))
            all_est_embeddings_er = [torch.zeros(self.er_batch_size, self.acmodel.semi_memory_size[j], device=self.device) \
                                     for j in range(self.num_agents)]

            all_actions_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_next_actions_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_broadcasts_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_next_broadcasts_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_rewards_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_agents_values_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)
            all_next_agents_values_er = torch.zeros((self.num_agents, self.er_batch_size), device=self.device)

            option_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                           for _ in range(self.num_agents)]
            action_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                           for _ in range(self.num_agents)]
            next_action_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                                for _ in range(self.num_agents)]
            broadcast_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                              for _ in range(self.num_agents)]
            next_broadcast_idxs = [torch.zeros(self.er_batch_size, device=self.device) \
                                   for _ in range(self.num_agents)]


            if self.acmodel.use_central_critic:
                sbs_coord = None
                update_critic_loss = 0
                update_value = 0
                update_value_loss = 0
            else:
                update_value = [0 for _ in range(self.num_agents)]
                update_value_loss = [0 for _ in range(self.num_agents)]
                update_critic_loss = [0 for _ in range(self.num_agents)]

            if self.acmodel.recurrent:
                memories = [exps[j].memory[inds] for j in range(self.num_agents)]

            update_entropy = [0 for _ in range(self.num_agents)]
            update_broadcast_entropy = [0 for _ in range(self.num_agents)]
            update_policy_loss = [0 for _ in range(self.num_agents)]
            update_broadcast_loss = [0 for _ in range(self.num_agents)]
            update_actor_loss = [0 for _ in range(self.num_agents)]

            for i in range(self.recurrence):
                if self.acmodel.use_central_critic:
                    sbs_coord = coord_exps[inds + i]

                for j in range(self.num_agents):
                    # Initialize memory

                    if self.acmodel.recurrent:
                        memory = memories[j]

                        sbs[j] = exps[j][inds + i]


                        # Compute loss
                        # if len(self.replayBuffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
                        #     return
                        # if not t % 100 == 0:  # only update every 100 steps
                        #     return

                        self.replay_sample_index = self.replayBuffer.make_index(sbs[j], self.er_batch_size)

                        policy_loss = 0.0
                        value_loss = 0.0
                        loss = 0.0
                        entropy = 0.0
                        broadcast_entropy = 0.0

                        # print('batch_size', self.batch_size, 'agent_obs', type(agent_obs[0]))

                        for ind, k in enumerate(
                                self.replay_sample_index):  # batch_size is the number os samples drawn from agent's buffer

                            agent_obs = sbs[j].obs[k]
                            agent_obs.image = agent_obs.image.unsqueeze(0)
                            agent_act = sbs[j].action[k].unsqueeze(0)
                            action_idxs[j][ind] = agent_act
                            agent_brd = sbs[j].broadcast[k].unsqueeze(0)
                            broadcast_idxs[j][ind] = agent_brd
                            agent_reward = sbs[j].reward_plus_broadcast_penalties[k].unsqueeze(0)
                            agent_next_obs = sbs[j].next_obs[k]
                            agent_next_obs.image = agent_next_obs.image.unsqueeze(0)
                            agent_advantage = sbs[j].advantage[k].unsqueeze(0)
                            agent_est_embedding = sbs[j].estimated_embedding[k].unsqueeze(0)
                            agent_done = sbs[j].mask[k].unsqueeze(0)
                            agent_memory = sbs[j].memory[k].unsqueeze(0)
                            option_in_use = sbs[j].current_options[k].unsqueeze(0)

                            option_idxs[j][ind] = option_in_use

                            if self.acmodel.recurrent:
                                if not self.acmodel.always_broadcast:
                                    act_dist, act_values, act_values_b, memory, _, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                                        agent_obs, agent_memory * (1. * agent_done), agent_index=j)
                                    next_act_dist, next_act_values, next_act_values_b, next_memory, _, next_broadcast_dist, next_embedding = self.acmodel.forward_agent_critic(
                                        agent_next_obs, agent_memory * (1. * agent_done), agent_index=j)
                                else:
                                    act_dist, act_values, memory, _, embedding = self.acmodel.forward_agent_critic(
                                        agent_obs,
                                        agent_memory * (1. * agent_done),
                                        agent_index=j)
                                    next_act_dist, next_act_values, next_memory, _, next_embedding = self.acmodel.forward_agent_critic(
                                        agent_next_obs,
                                        agent_memory * (1. * agent_done),
                                        agent_index=j)
                            else:
                                if not self.acmodel.always_broadcast:
                                    act_dist, act_values, act_values_b, _, _, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                                        agent_obs, agent_memory * (1. * agent_done), agent_index=j)
                                    next_act_dist, next_act_values, next_act_values_b, _, _, next_broadcast_dist, next_embedding = self.acmodel.forward_agent_critic(
                                        agent_next_obs, agent_memory * (1. * agent_done), agent_index=j)
                                else:
                                    act_dist, act_values, _, _, embedding = self.acmodel.forward_agent_critic(agent_obs,
                                                                                                              agent_index=j)
                                    next_act_dist, next_act_values, _, _, next_embedding = self.acmodel.forward_agent_critic(
                                        agent_next_obs,
                                        agent_index=j)


                            next_agent_act = next_act_dist.sample()[option_in_use].squeeze(1)

                            next_action_idxs[j][ind] = next_agent_act
                            entropy += act_dist.entropy().mean()
                            agent_act_log_probs = \
                                act_dist.log_prob(agent_act.view(-1, 1).repeat(1, self.num_options))[
                                    range(len(agent_act)), option_in_use]

                            policy_loss += -(agent_act_log_probs * agent_advantage).mean()

                            agent_values = act_values[range(len(agent_act)), option_in_use]
                            agent_next_values = next_act_values[range(len(agent_act)), option_in_use]

                            all_embeddings_er[j][ind] = embedding
                            all_next_embeddings_er[j][ind] = next_embedding
                            all_est_embeddings_er[j][ind] = agent_est_embedding
                            all_actions_er[j][ind] = agent_act
                            all_next_actions_er[j][ind] = next_agent_act
                            all_broadcasts_er[j][ind] = agent_brd
                            all_rewards_er[j][ind] = agent_reward
                            all_agents_values_er[j][ind] = agent_values
                            all_next_agents_values_er[j][ind] = agent_next_values

                            if not self.acmodel.use_central_critic:
                                delta = agent_reward + self.discount * agent_next_values - agent_values
                                # value_loss = (agent_values - sbs[j].returnn).pow(2).mean()
                                value_loss += delta.pow(2).mean()

                                update_value[j] += agent_values.mean().item()
                                update_value_loss[j] += value_loss.item()

                            if not self.acmodel.always_broadcast:
                                next_agent_brd = next_broadcast_dist.sample().squeeze()[next_agent_act]
                                next_broadcast_idxs[j][ind] = next_agent_brd
                                all_next_broadcasts_er[j][ind] = next_agent_brd

                                broadcast_entropy += broadcast_dist.entropy().mean()
                                broadcast_log_probs = broadcast_dist.log_prob(
                                    sbs[j].broadcast.view(-1, 1, 1).repeat(1, self.num_options, self.num_actions[j]))[
                                    range(len(agent_brd)), option_in_use, agent_act.long()]
                                broadcast_loss = -(broadcast_log_probs * agent_advantage).mean()

                                loss += policy_loss - self.entropy_coef * entropy \
                                        + broadcast_loss \
                                        - self.entropy_coef * broadcast_entropy
                            else:
                                loss += policy_loss - self.entropy_coef * entropy

                        # sample averages
                        policy_loss /= self.er_batch_size
                        loss /= self.er_batch_size
                        entropy /= self.er_batch_size
                        broadcast_entropy /= self.er_batch_size

                        if not self.acmodel.use_central_critic:
                            value_loss /= self.er_batch_size
                            update_critic_loss[j] += self.value_loss_coef * value_loss
                            update_value[j] /= self.er_batch_size
                            update_value_loss[j] /= self.er_batch_size

                        # Update batch values

                        update_entropy[j] += entropy.item()
                        if not self.acmodel.always_broadcast:
                            update_broadcast_entropy[j] += broadcast_entropy.item()
                            update_broadcast_loss[j] += broadcast_loss.item()

                        update_policy_loss[j] += policy_loss.item()
                        update_actor_loss[j] += loss

                if self.acmodel.use_central_critic:
                    # Collect agent embeddings

                    for ind, k in enumerate(self.replay_sample_index):
                        op_inds = []
                        ac_inds = []
                        next_ac_inds = []
                        brd_inds = []
                        next_brd_inds = []

                        for j in range(self.num_agents):
                            op_inds.append(option_idxs[j][ind].unsqueeze(0))
                            ac_inds.append(action_idxs[j][ind].unsqueeze(0))
                            next_ac_inds.append(next_action_idxs[j][ind].unsqueeze(0))
                            brd_inds.append(broadcast_idxs[j][ind].unsqueeze(0))
                            next_brd_inds.append(next_broadcast_idxs[j][ind].unsqueeze(0))

                            embeddings[j] = all_embeddings_er[j][ind]
                            next_embeddings[j] = all_next_embeddings_er[j][ind]

                            # Collect masked (coord) embedding


                            if self.acmodel.always_broadcast:
                                masked_embeddings[j] = all_embeddings_er[j][ind].unsqueeze(0)
                                estimated_embeddings[j] = all_embeddings_er[j][ind].unsqueeze(0)
                                next_estimated_embeddings[j] = all_next_embeddings_er[j][ind].unsqueeze(0)
                            else:
                                masked_embeddings[j] = all_broadcasts_er[j][ind].unsqueeze(0) * all_embeddings_er[j][
                                    ind].unsqueeze(0)
                                estimated_embeddings[j] = all_est_embeddings_er[j][ind].unsqueeze(0)

                                with torch.no_grad():
                                    next_estimated_embeddings[j] = (next_broadcast_idxs[j][ind] * next_embeddings[
                                        j]).unsqueeze(0) + ((1. - next_broadcast_idxs[j][ind]) * embeddings[j]).unsqueeze(0)


                        _, value_a_b, _ = self.acmodel.forward_central_critic(estimated_embeddings, op_inds,
                                                                              ac_inds, brd_inds,
                                                                              sbs_coord.memory[k].unsqueeze(0))
                        _, next_value_a_b, _ = self.acmodel.forward_central_critic(next_estimated_embeddings, op_inds,
                                                                                   next_ac_inds, next_brd_inds,
                                                                                   sbs_coord.memory[k].unsqueeze(0))

                        avg_value_loss = 0.0
                        for j in range(self.num_agents):
                            # avg_value_loss = avg_value_loss + (value_a_b - sbs[j].returnn).pow(2).mean()
                            # delta1 = all_rewards_er[j][ind] + self.discount * all_next_agents_values_er[j][ind] - all_agents_values_er[j][ind]
                            delta1 = all_rewards_er[j][ind] + self.discount * next_value_a_b - \
                                     value_a_b
                            avg_value_loss = avg_value_loss + delta1.pow(2).mean()
                        value_loss += avg_value_loss / self.num_agents

                        # update_value += coord_value_action.mean().item()
                        update_value += value_a_b.mean().item()

                    value_loss /= self.er_batch_size
                    update_critic_loss += self.value_loss_coef * value_loss
                    # print('update_critic_loss', update_critic_loss)
                    update_value /= self.er_batch_size
                    update_value_loss += value_loss.item()

            self.optimizer.zero_grad()

            # Actors back propagation

            for j in range(self.num_agents):

                update_entropy[j] /= self.recurrence  # recurrence_agents

                update_policy_loss[j] /= self.recurrence
                if not self.always_broadcast:
                    update_broadcast_entropy[j] /= self.recurrence
                    update_broadcast_loss[j] /= self.recurrence

                update_actor_loss[j] /= self.recurrence

                # update_values_b[j] /= self.recurrence

                update_actor_loss[j].backward(retain_graph=True)

            # Critic back propagation
            if self.acmodel.use_central_critic:
                update_value /= self.recurrence  # recurrence_coord
                update_value_loss /= self.recurrence
                update_critic_loss.backward()
            else:
                for j in range(self.num_agents):
                    update_value[j] /= self.recurrence  # recurrence_coord
                    update_value_loss[j] /= self.recurrence
                    update_critic_loss[j].backward(retain_graph=True)

            for name, param in self.acmodel.named_parameters():
                # print('name', name, 'param.data', param.data, 'param_grad', param.grad)
                if param.grad is None:
                    print('Grad_none', name)
            update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Log some values
        logs["entropy"] = update_entropy
        logs["broadcast_entropy"] = update_broadcast_entropy
        logs["value"] = update_value
        logs["policy_loss"] = update_policy_loss
        logs["broadcast_loss"] = update_broadcast_loss
        logs["value_loss"] = update_value_loss
        logs["grad_norm"] = update_grad_norm

        if self.acmodel.use_central_critic:
            logs["options"] = option_idxs
            logs["actions"] = action_idxs

        #print('maddpg_log_retun', logs["return_per_episode_with_broadcast_penalties"])

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = np.arange(0, self.num_frames, self.recurrence)
        return starting_indexes

    # def update(self, agents, t):
    #     if len(self.replayBuffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
    #         return
    #     if not t % 100 == 0:  # only update every 100 steps
    #         return
    #
    #     self.replay_sample_index = self.replayBuffer.make_index(self.batch_size)
    #     # collect replay sample from all agents
    #     obs_n = []
    #     obs_next_n = []
    #     act_n = []
    #     index = self.replay_sample_index
    #     for i in range(self.n):
    #         obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
    #         obs_n.append(obs)
    #         obs_next_n.append(obs_next)
    #         act_n.append(act)
    #     obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
    #
    #     # train q network
    #     num_sample = 1
    #     target_q = 0.0
    #     for i in range(num_sample):
    #         target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
    #         target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
    #         target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
    #     target_q /= num_sample
    #     q_loss = self.q_train(*(obs_n + act_n + [target_q]))
    #
    #     # train p network
    #     p_loss = self.p_train(*(obs_n + act_n))
    #
    #     self.p_update()
    #     self.q_update()
    #
    #     return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

