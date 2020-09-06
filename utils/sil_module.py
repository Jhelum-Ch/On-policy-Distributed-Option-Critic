import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import copy
#from collections import defaultdict
from utils.buffer import PrioritizedReplayBuffer
#from utils.misc import evaluate_actions_sil




# self-imitation learning
class sil_module:
    def __init__(self, config, env_dims, agents, num_options, gamma, ent_wt):
        #self.args = args
        self.config = config
        self.num_options = num_options
        self.gamma = gamma
        self.ent_wt = ent_wt
        self.env_dims = env_dims
        self.agents = agents
        self.num_frames = self.config.frames_per_proc * self.config.procs
        self.replay_buffer = PrioritizedReplayBuffer(self.config.env, self.config.buffer_length, self.config.num_agents, self.env_dims[0], \
                                              self.env_dims[1], self.env_dims[2], self.env_dims[3], \
                                                     self.config.sil_alpha, ep_buffer=False)
        self.episode_buffer = PrioritizedReplayBuffer(self.config.env, self.config.buffer_length, self.config.num_agents, self.env_dims[0], \
                                              self.env_dims[1], self.env_dims[2], self.env_dims[3], \
                                                     self.config.sil_alpha, ep_buffer=True)
        #self.acmodel = acmodel

        # super().__init__(config=config, env_dims=env_dims, num_agents=num_agents, envs=envs, acmodel=acmodel, \
        #                  sil_model=self.sil_model, replay_buffer=replay_buffer, no_sil=no_sil, \
        #                  num_frames_per_proc=num_frames_per_proc, discount=discount, lr=lr, gae_lambda=gae_lambda, \
        #                  entropy_coef=entropy_coef,
        #                  value_loss_coef=value_loss_coef, max_grad_norm=max_grad_norm, recurrence=recurrence,
        #                  preprocess_obss=preprocess_obss, \
        #                  reshape_reward=reshape_reward, broadcast_penalty=broadcast_penalty,
        #                  termination_reg=termination_reg, termination_loss_coef=termination_loss_coef)
        #



        # self.policies = [self.agents[i].policy for i in range(len(self.agents))]
        # self.critics = [self.agents[i].critic for i in range(len(self.agents))]
        # self.br_policies = [self.agents[i].br_policy for i in range(len(self.agents))]
        # self.termination = [self.agents[i].termination for i in range(len(self.agents))]
        #
        # self.target_policies = [self.agents[i].target_policy for i in range(len(self.agents))]
        # self.target_critics = [self.agents[i].target_critic for i in range(len(self.agents))]
        # self.target_br_policies = [self.agents[i].target_br_policy for i in range(len(self.agents))]
        # self.target_termination = [self.agents[i].target_termination for i in range(len(self.agents))]
        #
        # self.running_episodes = [[] for _ in range(self.config.num_processors)]
        #
        # self.policy_optimizers = [self.agents[i].policy_optimizer for i in range(len(self.agents))]
        # self.br_policy_optimizers = [self.agents[i].br_policy_optimizer for i in range(len(self.agents))]
        # self.critic_optimizers = [self.agents[i].critic_optimizer for i in range(len(self.agents))]
        # self.termination_optimizers = [self.agents[i].termination_optimizer for i in range(len(self.agents))]
        #
        # # self.buffer = PrioritizedReplayBuffer(self.buffer_length, self.config.num_agents,
        # #                          self.env_dims[0], self.env_dims[1], self.env_dims[2], self.env_dims[3],
        # #                          self.sil_alpha)
        #
        # # some other parameters...
        # self.total_steps = []
        # self.total_rewards = []

    # add the batch information into it...for each processor separately
    # def step(self, old_embeddings, obs, options, actions, broadcasts, betas, memories, cr_memory, rewards, next_obs, dones):
    #     for n in range(self.config.num_processors):
    #         obs_n = [obs[i][n] for i in range(len(obs))]
    #         if not all(o is None for o in old_embeddings):
    #             old_embeddings_n = [old_embeddings[i][n] for i in range(len(old_embeddings))]
    #         else:
    #             old_embeddings_n = old_embeddings
    #         options_n = [options[i][n] for i in range(len(options))]
    #         actions_n = [actions[i][n] for i in range(len(actions))]
    #         broadcasts_n = [broadcasts[i][n] for i in range(len(broadcasts))]
    #         betas_n = [betas[i][n] for i in range(len(betas))]
    #         memories_n = [memories[i][n] for i in range(len(memories))]
    #         cr_memory_n = [cr_memory[i][n] for i in range(len(cr_memory))]
    #         rewards_n = [rewards[i][n] for i in range(len(rewards))]
    #         next_obs_n = [next_obs[i][n] for i in range(len(next_obs))]
    #
    #         self.running_episodes[n].append([obs_n, old_embeddings_n, options_n, actions_n, broadcasts_n, betas_n, \
    #                                          memories_n, cr_memory_n, rewards_n, next_obs_n])
    #     # to see if can update the episode...push into priority buffer
    #
    #     for i in range(len(obs)):
    #         for n, done in enumerate(dones):
    #             if i == 0:
    #                 done['agent-'+str(i)] = True
    #             if done['agent-'+str(i)]:
    #                 self.update_buffer(self.running_episodes[n])
    #                 self.running_episodes[n] = []


    '''add the batch information into sil model...update buffer at every step'''
    def step(self, step_num, old_embeddings, obs, options, actions, broadcasts, betas, memories, cr_memory, rewards, br_rewards, next_obs, dones):
        self.running_episodes.append([obs, old_embeddings, options, actions, broadcasts, betas, \
                                             memories, cr_memory, rewards, br_rewards, next_obs])

        # push into priority buffer
        #obs = [torch.stack(obs[i]) for i in range(self.config.num_agents)]

        #if isinstance(options[0], np.ndarray):
        options = torch.tensor(options)
        options = options.view(self.config.num_agents, self.config.num_processors, -1)
        #options = [torch.tensor(options[i]) for i in range(self.config.num_agents)]

        #if isinstance(actions[0], np.ndarray):
        agents_ac_dims = []
        if self.config.env=='simple_speaker_listener': # it has mixed action space
            for i in range(len(actions)):
                agents_ac_dims.append(np.shape(actions[i])[1])
            #actions = torch.cat([torch.tensor(item) for item in actions], dim=1)
        else:
            actions = torch.tensor(actions)
            actions = actions.view(self.config.num_agents, self.config.num_processors, -1)
        #actions = [torch.tensor(actions[i]) for i in range(self.config.num_agents)]

        #if isinstance(broadcasts[0], np.ndarray):
        broadcasts = torch.tensor(broadcasts)
        broadcasts = broadcasts.view(self.config.num_agents, self.config.num_processors, -1)
        #broadcasts = [torch.tensor(broadcasts[i]) for i in range(self.config.num_agents)]

        #if isinstance(rewards[0], list):
        rewards = torch.tensor(rewards)
        rewards = rewards.view(self.config.num_agents, self.config.num_processors)
        #rewards = [torch.tensor(rewards[i]) for i in range(self.config.num_agents)]

        #next_obs = [torch.stack(next_obs[i]) for i in range(self.config.num_agents)]

        if not self.config.env == 'cleanup' and not self.config.env == 'harvest':
            dones = dones.reshape(self.config.num_agents, self.config.num_processors)


        # self.replay_buffer.push(obs, old_embeddings, options, actions, broadcasts, betas, memories, cr_memory, rewards, \
        #                  br_rewards, next_obs, dones)

        # update episode buffer if done. It needs to be done separately for each processor

        for n in range(self.config.num_processors):
            obs_n = [obs[i][n] for i in range(len(obs))]
            if not all(o is None for o in old_embeddings):
                old_embeddings_n = [old_embeddings[i][n] for i in range(len(old_embeddings))]
            else:
                old_embeddings_n = old_embeddings
            options_n = [options[i][n] for i in range(len(options))]
            actions_n = [actions[i][n] for i in range(len(actions))]
            broadcasts_n = [broadcasts[i][n] for i in range(len(broadcasts))]
            betas_n = [betas[i][n] for i in range(len(betas))]
            memories_n = [memories[i][n] for i in range(len(memories))]
            cr_memory_n = cr_memory[n]
            rewards_n = [rewards[i][n] for i in range(len(rewards))]
            br_rewards_n = [br_rewards[i][n] for i in range(len(br_rewards))]
            next_obs_n = [next_obs[i][n] for i in range(len(next_obs))]


            self.running_episodes[n].append([obs_n, old_embeddings_n, options_n, actions_n, broadcasts_n, betas_n, \
                                                    memories_n, cr_memory_n, rewards_n, br_rewards_n, next_obs_n])
        #for i in range(len(obs)):
        if self.config.env=='cleanup' or self.config.env=='harvest':
            done_list = [[dones[i]['agent-'+ str(j)] for j in range(self.config.num_agents)] for i in range(len(dones))]
        else:
            done_list = dones.reshape(self.config.num_processors, self.config.num_agents)
        for n, done in enumerate(done_list):
            # TODO: delete lines 132-133
            if True in done or step_num == self.config.episode_length-1:
                # update episode buffer if any one of the agent's episode has ended
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []

    # train the sil model...
    def train_sil_model(self):
        for n in range(self.config.n_episodes): # num_updates
            # sample from episode buffer
            obss, old_embeddings, opss, acss, brss, betass, memss, cr_memss, retss, br_retss, next_obss, doness, weightss, idxes = \
                self.sample_batch(self.config.batch_size, ep_buffer=True)
            # informed_ent_wt = [torch.ones(self.env_dims[2][i]) for i in range(self.config.num_agents)]
            # for i in range(self.config.num_agents):
            #     o_a_map = {k: [] for k in [str(item) for item in obss[i]]}
            #     for j in range(len(obss[i])):
            #         o_a_map[obss[i][j]].append(acss[i][j])
            #     #informed_ent_wt[i] -=
            mean_advs, mean_br_advs, num_valid_sampless = [], [], []

            if obss is not None:
                for agent_id in range(len(obss)):
                    retss[agent_id] = retss[agent_id].clone().unsqueeze(1)
                    br_retss[agent_id] = br_retss[agent_id].clone().unsqueeze(1)
                    torch_weight = torch.tensor(weightss[agent_id], dtype=torch.float32).unsqueeze(1)

                    # TODO: fix lines 153-159
                    print('SIL')
                    max_nlogp = torch.tensor(np.ones((len(idxes), 1)) * self.config.max_nlogp, dtype=torch.float32)
                    # if self.args.cuda:
                    #     obss = obss.cuda()
                    #     actions = actions.cuda()
                    #     returns = returns.cuda()
                    #     weights = weights.cuda()
                    #     max_nlogp = max_nlogp.cuda()

                    # start to next...
                    prediction, _, _ = self.agents[agent_id].forward(obss[agent_id], memss[agent_id])
                    value, _, _ = self.agents[agent_id].critic_forward(obss, acss, brss, cr_memss, old_embeddings,
                                                                 init_ac=False)
                    value = value.view(-1, self.num_options, self.config.num_agents)
                    if memss[agent_id] is None:
                        prediction, embedding, agent_memory = self.agents[agent_id].forward(obss[agent_id])
                    else:
                        prediction, embedding, agent_memory = self.agents[agent_id].forward(obss[agent_id], memss[agent_id])


                    op_indx = [(opss[agent_id][j, :] == 1.).nonzero()[0] for j in range(opss[agent_id].shape[0])]
                    op_indx = torch.stack(op_indx)

                    action_log_probs = torch.stack([prediction['log_pi'][j, op_indx.squeeze(1)[j], :] \
                                for j in range(prediction['log_pi'].shape[0])])


                    pi_prob = torch.stack([prediction['pi'][j, op_indx.squeeze(1)[j], :] \
                                for j in range(prediction['pi'].shape[0])])

                    print('no-sil', self.config.no_sil, 'informed', self.config.informed_exploration)
                    if self.config.informed_exploration:
                        pi = self.agents[agent_id].policy(obss[agent_id])
                        pi = pi.view(-1, self.config.num_options, self.agents[agent_id].num_ac)
                        pi = F.softmax(-pi, dim=-1)
                        pi_inf_exp = torch.stack([pi[j, op_indx.squeeze(1)[j], :] \
                                    for j in range(pi.shape[0])])
                        # import pdb;
                        # pdb.set_trace()
                        #pi_inf_exp = pi_inf_exp.detach()
                        pi_inf_exp = (pi_inf_exp * pi_prob) / torch.sum(pi_inf_exp * pi_prob, dim=1).unsqueeze(1)


                    #TODO:now pi_inf_exp becoming 0.5 everywhere (highest possible entropy).\
                    # Check if we should modify this

                        pi_dist = torch.distributions.Categorical(probs=pi_inf_exp)
                    else:
                        pi_dist = torch.distributions.Categorical(probs=pi_prob)

                    pi_entropy = pi_dist.entropy()

                    beta = torch.gather(prediction['beta'], 1, op_indx.long())

                    value = torch.stack([value[j, op_indx.squeeze(1)[j], agent_id].unsqueeze(0) \
                                         for j in range(value.shape[0])])
                    # TODO: add entropy term for informed exploration (see overleaf)

                    action_log_probs = -action_log_probs
                    clipped_nlogp_ac = torch.min(action_log_probs, max_nlogp)


                    # process returns

                    advantages = retss[agent_id] - value
                    advantages = advantages.detach()


                    masks = (advantages.cpu().numpy() > 0).astype(np.float32)

                    # get the num of vaild samples
                    num_valid_samples = np.sum(masks)
                    num_samples = np.max([num_valid_samples, self.config.mini_batch_size])

                    # process the mask
                    masks = torch.tensor(masks, dtype=torch.float32)

                    # TODO: fix cuda statements
                    # if self.args.cuda:
                    #     masks = masks.cuda()

                    # clip the advantages...
                    clipped_advantages = torch.clamp(advantages, 0, self.config.sil_clip)


                    mean_adv = torch.sum(clipped_advantages) / num_samples
                    mean_adv = mean_adv.item()



                    # start to get the action loss...
                    action_loss = torch.sum(clipped_advantages * torch_weight * clipped_nlogp_ac) / num_samples
                    beta_loss = torch.sum(beta * clipped_advantages * torch_weight) / num_samples

                    pi_entropy_reg = torch.sum(torch_weight * pi_entropy * masks) / num_samples

                    policy_loss = action_loss - pi_entropy_reg * self.ent_wt


                    # start to process the value loss..
                    # get the value loss

                    delta = torch.clamp(value - retss[agent_id], -self.config.sil_clip, 0) * masks
                    delta = delta.detach()
                    critic_loss = torch.sum(torch_weight * value * delta) / num_samples


                    self.agents[agent_id].policy_optimizer.zero_grad()
                    #import pdb; pdb.set_trace()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy.parameters(), self.agents[agent_id].gradient_clip)
                    self.agents[agent_id].policy_optimizer.step()



                    self.agents[agent_id].termination_optimizer.zero_grad()
                    beta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agents[agent_id].termination.parameters(), self.agents[agent_id].gradient_clip)
                    self.agents[agent_id].termination_optimizer.step()

                    self.agents[agent_id].critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agents[agent_id].critic.parameters(), self.agents[agent_id].gradient_clip)
                    self.agents[agent_id].critic_optimizer.step()

                    # update the priorities
                    self.episode_buffer.update_priorities(idxes, clipped_advantages.squeeze(1).cpu().numpy())
                    mean_advs.append(mean_adv)
                    num_valid_sampless.append(num_valid_samples)



                    if not self.config.always_br:
                        #if self.config.discrete_broadcast:
                        broadcast_log_probs = torch.stack([prediction['log_br_pi'][j, op_indx.squeeze(1)[j], :] \
                            for j in range(prediction['log_br_pi'].shape[0])])

                        br_pi_prob = torch.stack([prediction['br_pi'][j, op_indx.squeeze(1)[j], :] \
                            for j in range(prediction['br_pi'].shape[0])])
                        br_pi_dist = torch.distributions.Categorical(probs=br_pi_prob)

                        br_pi_entropy = br_pi_dist.entropy()

                        broadcast_log_probs = -broadcast_log_probs
                        clipped_nlogp_br = torch.min(broadcast_log_probs, max_nlogp)
                        br_advantages = br_retss[agent_id] - value
                        br_advantages = br_advantages.detach()

                        clipped_br_advantages = torch.clamp(br_advantages, 0, self.config.sil_clip)

                        mean_br_adv = torch.sum(clipped_br_advantages) / num_samples
                        mean_br_adv = mean_br_adv.item()

                        broadcast_loss = torch.sum(clipped_br_advantages * torch_weight * clipped_nlogp_br) / num_samples
                        br_pi_entropy_reg = torch.sum(torch_weight * br_pi_entropy * masks) / num_samples
                        br_policy_loss = broadcast_loss - br_pi_entropy_reg * self.ent_wt

                        self.agents[agent_id].br_policy_optimizer.zero_grad()
                        br_policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agents[agent_id].br_policy.parameters(),
                                                       self.agents[agent_id].gradient_clip)
                        self.agents[agent_id].br_policy_optimizer.step()

                        mean_br_advs.append(mean_br_adv)

        return mean_advs, mean_br_advs, num_valid_sampless

    def train_sil_teamgrid(self, experience, acmodel):
        rmsprop_alpha = 0.99
        rmsprop_eps = 1e-5

        self.optimizer = torch.optim.RMSprop(acmodel.parameters(), self.config.lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)
        # Collect experiences

        coord_exps, exps, logs = experience

        # Compute starting indexes

        inds = self._get_starting_indexes()
        max_nlogp = torch.tensor(np.ones((len(inds), 1)) * self.config.max_nlogp, dtype=torch.float32)

        # Initialize containers for actors info

        sbs_coord = None
        sbs = [None for _ in range(self.config.num_agents)]
        embeddings = [None for _ in range(self.config.num_agents)]
        masked_embeddings = [None for _ in range(self.config.num_agents)]
        estimated_embeddings = [None for _ in range(self.config.num_agents)]

        acss1 = [None for _ in range(self.config.num_agents)]
        opss1 = [None for _ in range(self.config.num_agents)]
        brdss1 = [None for _ in range(self.config.num_agents)]

        # last_broadcasts     = [None for _ in range(self.num_agents)]
        # last_masked_embeddings = [None for _ in range(self.num_agents)]

        if acmodel.recurrent:
            memories = [exps[j].memory[inds] for j in range(self.config.num_agents)]
            # coordinator_memory = coord_exps.memory[inds]

        update_entropy = [0 for _ in range(self.config.num_agents)]
        update_broadcast_entropy = [0 for _ in range(self.config.num_agents)]
        update_policy_loss = [0 for _ in range(self.config.num_agents)]
        update_broadcast_loss = [0 for _ in range(self.config.num_agents)]
        update_actor_loss = [0 for _ in range(self.config.num_agents)]
        # update_agent_critic_loss = [0 for _ in range(self.num_agents)]

        # Initialize scalars for central-critic info

        update_value = 0
        update_values = [0 for _ in range(self.config.num_agents)]
        # update_values_b = [0 for _ in range(self.num_agents)]
        update_value_loss = 0
        # update_values_loss = [0 for _ in range(self.num_agents)]
        # update_values_b_loss = [0 for _ in range(self.num_agents)]
        # update_agent_critic_loss = [0 for _ in range(self.num_agents)]
        # update_agent_critic_loss_b = [0 for _ in range(self.num_agents)]
        update_critic_loss = 0
        rets = [0 for _ in range(self.config.num_agents)]
        br_rets = [0 for _ in range(self.config.num_agents)]

        mean_advs, mean_br_advs, num_valid_sampless = [], [], []

        # Feed experience to model with gradient-tracking on a single process

        for i in range(self.config.sil_num_batches):  # recurrence_agents
            # print('i',i,'self.recurrence', self.recurrence)
            if acmodel.use_central_critic:
                sbs_coord = coord_exps[inds + i]


            for j in range(self.config.num_agents):

                # Create a sub-batch of experience

                sbs[j] = exps[j][inds + i]

                # Keep the components of sbs buffer where the advantage is positive.
                adv1 = []
                memories1 = []
                obs1 = []
                mask1 = []
                acs1 = []
                ops1  = []
                brs1 = []
                idxs = []
                for idx, l in enumerate(range(len(sbs[j].advantage))):
                    if sbs[j].advantage[l] > 0.0:
                        idxs.append(idx)
                        adv1.append(sbs[j].advantage[l])
                        memories1.append(memories[j][l])
                        obs1.append(sbs[j].obs.image[l])
                        mask1.append(sbs[j].mask[l])
                        acs1.append(sbs[j].action[l])
                        ops1.append(sbs[j].option[l])
                        brs1.append(sbs[j].broadcast[l])
                    else: #replace with zeros
                        adv1.append(torch.zeros_like(sbs[j].advantage[l]))
                        memories1.append(torch.zeros_like(memories[j][l]))
                        obs1.append(torch.zeros_like(sbs[j].obs.image[l]))
                        mask1.append(torch.zeros_like(sbs[j].mask[l]))
                        acs1.append(torch.zeros_like(sbs[j].action[l]))
                        ops1.append(torch.zeros_like(sbs[j].option[l]))
                        brs1.append(torch.zeros_like(sbs[j].broadcast[l]))


                obs1 = torch.stack(obs1)
                obs1 = {'image' : obs1}

                adv1 = torch.tensor(adv1)

                memories1 = torch.stack(memories1)
                mask1 = torch.tensor(mask1).unsqueeze(1)

                acs1 = torch.tensor(acs1)
                acss1[j] = acs1

                ops1 = torch.tensor(ops1)
                opss1[j] = ops1

                brs1 = torch.tensor(brs1)
                brdss1[j] = brs1




                # exp_dict = {str(ob):[] for ob in sbs[j].obs.image} #defaultdict()
                #
                # for l, ob in enumerate(sbs[j].obs.image):
                #     exp_dict[str(ob)].append((sbs[j].action[l], sbs[j].option[l], sbs[j].broadcast[l]))



                # Actor forward propagation

                # print('i', i, 'inds + i', len(inds + i), 'sbs_em', sbs[j].embeddings.size())

                if len(obs1) > 0: # if any component of sbs[j].advantage is positive

                    if not acmodel.always_broadcast:
                        act_mlp, act_dist, _, _, memory, term_dist, broadcast_dist, embedding = acmodel.forward_agent_critic(
                            obs1, \
                            memories1 * \
                            mask1, agent_index=j, sil_module=True)
                    else:
                        act_mlp, act_dist, _, memory, term_dist, embedding = acmodel.forward_agent_critic(
                            obs1, \
                            memories1 * \
                            mask1, agent_index=j, sil_module=True)

                    if self.config.informed_exploration:
                        ac_prob = torch.bincount(sbs[j].action, minlength=acmodel.num_actions).float()
                        pi_inf_exp = F.softmax(-ac_prob, dim=-1).unsqueeze(0)
                        pi_inf_exp_categorical = torch.distributions.Categorical(probs=pi_inf_exp)


                else: # use sbs[j] entirely. In this case SIL will have no effect
                    if not acmodel.always_broadcast:
                        act_mlp, act_dist, _, _, memory, term_dist, broadcast_dist, embedding = acmodel.forward_agent_critic(
                            sbs[j].obs, \
                            memories[j] * \
                            sbs[j].mask, agent_index=j, sil_module=True)
                    else:
                        act_mlp, act_dist, _, memory, term_dist, embedding = acmodel.forward_agent_critic(
                            sbs[j].obs, \
                            memories[j] * \
                            sbs[j].mask, agent_index=j, sil_module=True)

                    if self.config.informed_exploration:
                        ac_prob = torch.bincount(sbs[j].action, minlength=acmodel.num_actions).float()
                        pi_inf_exp = F.softmax(-ac_prob, dim=-1).unsqueeze(0)
                        pi_inf_exp_categorical = torch.distributions.Categorical(probs=pi_inf_exp)


                # compute MC returns
                rets[j], br_rets[j] = self.monte_carlo_return(sbs[j].reward, sbs[j].broadcast, sbs[j].mask, self.config.discount)


                # Collect agent embedding

                embeddings[j] = embedding

                # Collect masked (coord) embedding

                # if self.acmodel.use_broadcasting and self.acmodel.always_broadcast:
                if acmodel.always_broadcast:
                    # import pdb;
                    # pdb.set_trace()
                    masked_embeddings[j] = embedding
                    estimated_embeddings[j] = embedding
                else:
                    masked_embeddings[j] = sbs[j].broadcast.unsqueeze(1) * embedding

                    if len(obs1) > 0:
                        agent_est_embed = copy.deepcopy(sbs[j].estimated_embedding)
                        for idx in range(len(agent_est_embed)):
                            if idx not in idxs:
                                agent_est_embed[idx] = torch.zeros_like(sbs[j].estimated_embedding[idx])
                        #agent_est_embed = torch.tensor(agent_est_embed)
                        estimated_embeddings[j] = agent_est_embed
                    else:
                        estimated_embeddings[j] = sbs[j].estimated_embedding


            # Central-critic forward propagation
            if len(obs1) > 0:
                #option_idxs = [ops1[j] for j in range(self.config.num_agents)]
                #action_idxs = [sbs[j].action for j in range(self.config.num_agents)]

                #if acmodel.use_broadcasting:
                    #broadcast_idxs = [sbs[j].broadcast for j in range(self.config.num_agents)]
                coord_mem = copy.deepcopy(sbs_coord.memory)
                for idx in range(len(coord_mem)):
                    if idx not in idxs:
                        coord_mem[idx] = torch.zeros_like(sbs_coord.memory[idx])


                _, value_a_b, _ = acmodel.forward_central_critic(estimated_embeddings, opss1, acss1,
                                                                      brdss1, coord_mem)
            else:
                option_idxs = [sbs[j].current_options for j in range(self.config.num_agents)]
                action_idxs = [sbs[j].action for j in range(self.config.num_agents)]

                if acmodel.use_broadcasting:
                    broadcast_idxs = [sbs[j].broadcast for j in range(self.config.num_agents)]

                _, value_a_b, _ = acmodel.forward_central_critic(estimated_embeddings, option_idxs, action_idxs,
                                                                 broadcast_idxs, sbs_coord.memory)

            value_losses = 0
            for j in range(self.config.num_agents):
                # Actor losses

                entropy = act_dist.entropy().mean()
                if self.config.informed_exploration:
                    informed_exploration_entropy = pi_inf_exp_categorical.entropy().mean()
                    informed_exploration_entropy = torch.tensor(informed_exploration_entropy, requires_grad=True)
                    entropy += informed_exploration_entropy
                #print('entropy', entropy, 'informed_entropy', informed_exploration_entropy)

                # if self.acmodel.use_broadcasting and not self.acmodel.always_broadcast:
                if not acmodel.always_broadcast:
                    broadcast_entropy = broadcast_dist.entropy().mean()
                    if acmodel.use_teamgrid:
                        broadcast_log_probs = broadcast_dist.log_prob(
                            sbs[j].broadcast.view(-1, 1, 1).repeat(1, self.config.num_options,
                                                                   acmodel.num_actions))[
                            range(sbs[j].broadcast.shape[0]), sbs[j].current_options, sbs[j].action.long()]
                        clipped_nlogp_br = torch.min(broadcast_log_probs, max_nlogp)

                        br_advantages = br_rets[j] - value_a_b
                        br_advantages = br_advantages.detach()

                    else:
                        pass
                        # broadcast_log_probs = broadcast_dist.log_prob(
                        #     sbs[j].broadcast.view(-1, 1, 1).repeat(1, self.num_options, self.num_actions[j]))[
                        #     range(sbs[j].broadcast.shape[0]), sbs[j].current_options, sbs[j].action.long()]

                    broadcast_loss = -(broadcast_log_probs * (sbs[j].value_swa - sbs[j].value_sw)).mean()
                    # broadcast_loss = -(broadcast_log_probs * (sbs[j].value_swa_b - sbs[j].value_sw_b)).mean()
                    # broadcast_loss = -(broadcast_log_probs * (sbs_coord.value_swa - sbs_coord.value_sw)).mean()

                act_log_probs = act_dist.log_prob(sbs[j].action.view(-1, 1).repeat(1, self.num_options))[
                    range(sbs[j].action.shape[0]), sbs[j].current_options]

                advantages = rets[j] - value_a_b
                advantages = advantages.detach()

                clipped_nlogp_ac = torch.min(act_log_probs, max_nlogp)
                # clip the advantages...
                clipped_advantages = torch.clamp(advantages, 0, self.config.sil_clip)


                # process returns
                if acmodel.always_broadcast:
                    masks = (advantages.cpu().numpy() > 0).astype(np.float32)

                    # get the num of vaild samples
                    num_valid_samples = np.sum(masks)
                    num_samples = np.max([num_valid_samples, self.config.mini_batch_size])

                    # process the mask
                    masks = torch.tensor(masks, dtype=torch.float32)

                    # TODO: fix cuda statements
                    # if self.args.cuda:
                    #     masks = masks.cuda()


                    mean_adv = torch.sum(clipped_advantages) / num_samples
                    mean_adv = mean_adv.item()
                    mean_advs.append(mean_adv)
                else:
                    masks = (br_advantages.cpu().numpy() > 0).astype(np.float32)

                    # get the num of vaild samples
                    num_valid_samples = np.sum(masks)
                    num_samples = np.max([num_valid_samples, self.config.mini_batch_size])

                    # process the mask
                    masks = torch.tensor(masks, dtype=torch.float32)

                    # TODO: fix cuda statements

                    # clip the advantages...
                    clipped_br_advantages = torch.clamp(br_advantages, 0, self.config.sil_clip)

                    mean_adv = torch.sum(clipped_advantages) / num_samples
                    mean_adv = mean_adv.item()
                    mean_advs.append(mean_adv)
                    mean_br_adv = torch.sum(clipped_br_advantages) / num_samples
                    mean_br_adv = mean_br_adv.item()

                num_valid_sampless.append(num_valid_samples)

                # start to get the action loss...
                # TODO: here I am setting torch_weight=1.0 so that it is not really a priority buffer.
                #  Q: Is it necessary to make it a priority buffer?
                policy_loss = torch.sum(clipped_advantages * clipped_nlogp_ac) / num_samples


                # policy_loss = -(act_log_probs * (
                #         sbs[j].value_swa - sbs[j].value_sw)).mean()  # the second term should be coordinator value

                # policy_loss = (act_mlp.view(-1,1,1)[sbs[j].action.long()].squeeze() * (sbs[j].value_swa - sbs[j].value_sw)).mean() #doc-ml

                term_prob = term_dist.probs[range(sbs[j].action.shape[0]), sbs[j].current_options]
                #termination_loss = (term_prob * (sbs[j].advantage + self.termination_reg)).mean()
                termination_loss = torch.sum(term_prob * clipped_advantages) / num_samples
                # termination_loss = (term_prob * (sbs_coord.advantage + self.termination_reg)).mean()

                if acmodel.use_broadcasting and not acmodel.always_broadcast:
                    loss = policy_loss \
                           - self.config.entropy_coef * entropy \
                           + broadcast_loss \
                           - self.config.entropy_coef * broadcast_entropy \
                           + self.config.termination_loss_coef * termination_loss
                else:
                    loss = policy_loss \
                           - self.config.entropy_coef * entropy \
                           + self.config.termination_loss_coef * termination_loss

                # Update batch values
                update_entropy[j] += entropy.item()
                # if self.acmodel.use_broadcasting and not self.acmodel.always_broadcast:
                if not acmodel.always_broadcast:
                    update_broadcast_entropy[j] += broadcast_entropy.item()
                    update_broadcast_loss[j] += broadcast_loss.item()
                update_policy_loss[j] += policy_loss.item()

                update_actor_loss[j] += loss


                pi_entropy_reg = torch.sum(entropy * masks) / num_samples

                policy_loss = policy_loss - pi_entropy_reg * self.config.entropy_coef
                update_actor_loss[j]

                # start to process the value loss..
                # get the value loss

                delta = torch.clamp(value_a_b - rets[j], -self.config.sil_clip, 0) * masks
                delta = delta.detach()
                value_losses += torch.sum(value_a_b * delta) / num_samples

            value_loss = value_losses / self.config.num_agents

            # value_losses = 0
            # for j in range(self.config.num_agents):
            #     value_losses = value_losses + (value_a_b - sbs[j].target).pow(2).mean()
            # value_loss = value_losses / self.config.num_agents

            # Update batch values

            # update_value += coord_value_action.mean().item()
            update_value += value_a_b.mean().item()
            update_value_loss += value_loss.item()
            update_critic_loss += self.config.value_loss_coef * value_loss


        # Re-initialize gradient buffers

        self.optimizer.zero_grad()
        # self.optimizer_critic.zero_grad()

        # Actors back propagation

        for j in range(self.config.num_agents):
            if not acmodel.use_teamgrid:
                # reduce lr for agent 1 (movable)
                # print('j', j, 'self.lr', self.lr)
                if j != 0:
                    lr = 0.1 * self.lr
                    self.optimizer = torch.optim.RMSprop(acmodel.parameters(), lr,
                                                         alpha=self.config.rmsprop_alpha, eps=self.config.rmsprop_eps)
                    self.optimizer.zero_grad()
                    # print('j', j, 'self.lr', self.lr, 'lr', lr)
            update_entropy[j] /= self.config.sil_num_batches  # recurrence_agents

            update_policy_loss[j] /= self.config.sil_num_batches
            if not self.config.use_always_broadcast:
                update_broadcast_entropy[j] /= self.config.sil_num_batches
                update_broadcast_loss[j] /= self.config.sil_num_batches

            update_actor_loss[j] /= self.config.sil_num_batches
            update_values[j] /= self.config.sil_num_batches
            # update_values_b[j] /= self.recurrence

            update_actor_loss[j].backward(retain_graph=True)

            # TODO: do we need the following back prop?
            # update_agent_critic_loss[j].backward(retain_graph=True)
            # update_agent_critic_loss_b[j].backward(retain_graph=True)

        # Critic back propagation

        update_value /= self.config.sil_num_batches  # recurrence_coord
        update_value_loss /= self.config.sil_num_batches

        # update_critic_loss.backward(retain_graph=True)
        update_critic_loss /= self.config.sil_num_batches
        update_critic_loss.backward()

        # Learning step

        for name, param in acmodel.named_parameters():
            # print('name', name) #'param_data', param.data, 'param_grad', param.grad)
            if param.grad is None:
                print('Grad_none', name)

        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(acmodel.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return mean_advs, mean_br_advs, num_valid_sampless

    # update episode buffer
    def update_buffer(self, trajectory): # all agents' trajectories #TODO: fix it
        #traj_to_be_added = np.copy(trajectory)

        positive_reward = False



        # Keep the trajectory if any one of the agents have strictly positive reward
        for (ob, old_em, op, ac, br, beta, mem, cr_mem, r, br_r, next_o) in trajectory:
            for i in range(self.config.num_agents):
                if r[i] >= 0 or br_r[i] >= 0:
                    positive_reward = True
                    break
            else:
                continue
            break

        #TODO: in original SIL code, eoisode is added only when r > 0. What if r < 0 always? Then mothing is
        # added to episode buffer!

        #if positive_reward:
        self.add_episode(trajectory)
        self.total_steps.append(len(trajectory))
        self.total_rewards.append(np.sum([np.mean(x[8]) for x in trajectory]))  # add rewards

        while np.sum(self.total_steps) > self.config.buffer_length and len(self.total_steps) > 1:
            self.total_steps.pop(0)
            self.total_rewards.pop(0)

    # TODO: make it for all agents
    def add_episode(self, trajectory):
        obs = []
        old_ems = []
        options = []
        actions = []
        broadcasts = []
        betas = []
        mems = []
        cr_mems = []
        rewards = []
        br_rewards = []
        next_obs = []
        dones = []

        for (ob, old_em, op, ac, br, beta, mem, cr_mem, r, br_r, next_o) in trajectory:
            obs.append(ob)
            old_ems.append(old_em)
            options.append(op)
            actions.append(ac)
            broadcasts.append(br)
            betas.append(beta)
            mems.append(mem)
            rewards.append(np.sign(r))
            br_rewards.append(np.sign(br_r))
            next_obs.append(next_o)
            dones.append([False]*self.config.num_agents)
            cr_mems.append(cr_mem)
        dones[len(dones) - 1] = [True]*self.config.num_agents

        # raw (monte carlo) return from the environment
        returns, br_returns = self.monte_carlo_return(rewards, broadcasts, dones, self.gamma)

        for (ob, old_em, op, ac, br, beta, mem, cr_mem, R, br_R, next_o, done) in zip(obs, old_ems, options, actions, \
                                                            broadcasts, betas, mems, cr_mems, returns, br_returns, next_obs, dones):
            self.episode_buffer.push(ob, old_em, op, ac, br, beta, mem, cr_mem, R, br_R, next_o, done)


    def fn_reward(self, reward):
        return np.sign(reward)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0

    def num_episodes(self):
        return len(self.total_rewards)

    def num_steps(self):
        return len(self.replay_buffer)

    def sample_batch(self, batch_size, ep_buffer):
        if ep_buffer:
            if len(self.episode_buffer) > 1: # TODO: make it a larger number like 100 while training agents
                batch_size = min(batch_size, len(self.episode_buffer))
                return self.episode_buffer.sample(batch_size, self.config.num_agents, beta=self.config.sil_beta)
            else:
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None
        else:
            if len(self.replay_buffer) > 1: #> 100:
                batch_size = min(batch_size, len(self.replay_buffer))
                return self.replay_buffer.sample(batch_size, self.config.num_agents, beta=self.config.sil_beta)
            else:
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None

    def monte_carlo_return(self, rewards, broadcasts, dones, gamma):
        discounted_r = []
        discounted_br_r = []
        r = 0 #[0 for _ in range(self.config.num_agents)]
        br_r = 0 #[0 for _ in range(self.config.num_agents)]


        for reward, broadcast, done in zip(rewards.numpy()[::-1], broadcasts.numpy()[::-1], dones.numpy()[::-1]):
            r = reward + gamma * r
            discounted_r.append(r)
            if not self.config.use_always_broadcast:
                br_r = reward + broadcast * self.config.br_penalty + gamma * br_r
                discounted_br_r.append(br_r)
        return torch.tensor(discounted_r[::-1]), torch.tensor(discounted_br_r[::-1])
        # for reward, broadcast, done in zip(rewards[::-1], broadcasts[::-1], dones[::-1]):
        #     # shared reward
        #     total_rew = np.sum(reward)
        #     reward = [total_rew for _ in range(self.config.num_agents)]
        #     r = [reward[i] + gamma * r[i] * (1. - done[i]) \
        #          for i in range(self.config.num_agents)]
        #     br_r = [reward[i] + broadcast[i][-1] * self.config.br_penalty + gamma * br_r[i] * (1. - done[i]) \
        #          for i in range(self.config.num_agents)]
        #     discounted_r.append(r)
        #     discounted_br_r.append(br_r)
        # return discounted_r[::-1], discounted_br_r[::-1]

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

        starting_indexes = np.arange(0, self.num_frames, self.config.sil_num_batches)
        return starting_indexes