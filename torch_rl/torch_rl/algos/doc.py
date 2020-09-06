import numpy as np
import torch
import torch.nn.functional as F
import copy
from utils.sil_module import sil_module

from torch_rl.algos.base import BaseAlgo


class DOCAlgo(BaseAlgo):
    """The class for the Distributed Option-Critic algorithm."""


    def __init__(self, config=None, env_dims=None, num_agents=None, envs=None, acmodel=None, replay_buffer=None, \
                 no_sil=None, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence = 4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, num_options=3,
                 termination_loss_coef=0.5, termination_reg=0.01, reshape_reward=None, always_broadcast = False, \
                 broadcast_penalty=-0.01):

        num_frames_per_proc = num_frames_per_proc or 8

        self.config = config
        self.env_dims = env_dims

        self.sil_model = sil_module(self.config, self.env_dims, acmodel, self.config.num_options, self.config.discount, \
                                    self.config.entropy_coef)


        super().__init__(config=config, env_dims=env_dims, num_agents=num_agents, envs=envs, acmodel=acmodel, \
                         sil_model=self.sil_model, replay_buffer=replay_buffer, no_sil=no_sil,\
                         num_frames_per_proc=num_frames_per_proc, discount=discount, lr=lr, gae_lambda=gae_lambda, \
                         entropy_coef=entropy_coef,
         value_loss_coef=value_loss_coef, max_grad_norm=max_grad_norm, recurrence=recurrence, preprocess_obss=preprocess_obss, \
                         reshape_reward=reshape_reward, broadcast_penalty=broadcast_penalty,
         termination_reg=termination_reg, termination_loss_coef=termination_loss_coef)

        self.rmsprop_alpha = rmsprop_alpha
        self.rmsprop_eps = rmsprop_eps
        self.lr = lr

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), self.lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)


    def update_parameters(self):

        # Collect experiences

        coord_exps, exps, logs = self.collect_experiences()
        # exps, logs = self.collect_experiences()

        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize containers for actors info

        sbs_coord = None
        sbs = [None for _ in range(self.num_agents)]
        embeddings = [None for _ in range(self.num_agents)]
        masked_embeddings = [None for _ in range(self.num_agents)]
        estimated_embeddings = [None for _ in range(self.num_agents)]

        # last_broadcasts     = [None for _ in range(self.num_agents)]
        # last_masked_embeddings = [None for _ in range(self.num_agents)]

        if self.acmodel.recurrent:
            memories = [exps[j].memory[inds] for j in range(self.num_agents)]
            # coordinator_memory = coord_exps.memory[inds]

        update_entropy = [0 for _ in range(self.num_agents)]
        update_broadcast_entropy = [0 for _ in range(self.num_agents)]
        update_policy_loss = [0 for _ in range(self.num_agents)]
        update_broadcast_loss = [0 for _ in range(self.num_agents)]
        update_actor_loss = [0 for _ in range(self.num_agents)]
        # update_agent_critic_loss = [0 for _ in range(self.num_agents)]

        # Initialize scalars for central-critic info

        update_value = 0
        update_values = [0 for _ in range(self.num_agents)]
        # update_values_b = [0 for _ in range(self.num_agents)]
        update_value_loss = 0
        # update_values_loss = [0 for _ in range(self.num_agents)]
        # update_values_b_loss = [0 for _ in range(self.num_agents)]
        # update_agent_critic_loss = [0 for _ in range(self.num_agents)]
        # update_agent_critic_loss_b = [0 for _ in range(self.num_agents)]
        update_critic_loss = 0

        # Feed experience to model with gradient-tracking on a single process

        for i in range(self.recurrence):  # recurrence_agents
            # print('i',i,'self.recurrence', self.recurrence)
            if self.acmodel.use_central_critic:
                sbs_coord = coord_exps[inds + i]

            for j in range(self.num_agents):

                # Create a sub-batch of experience

                sbs[j] = exps[j][inds + i]

                # Actor forward propagation

                # print('i', i, 'inds + i', len(inds + i), 'sbs_em', sbs[j].embeddings.size())

                if not self.acmodel.always_broadcast:
                    act_mlp, act_dist, _, _, memory, term_dist, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                        sbs[j].obs, \
                        memories[j] * \
                        sbs[j].mask, agent_index=j, sil_module=False)
                else:
                    act_mlp, act_dist, _, memory, term_dist, embedding = self.acmodel.forward_agent_critic(
                        sbs[j].obs, \
                        memories[j] * \
                        sbs[j].mask, agent_index=j, sil_module=False)

                # Actor losses

                entropy = act_dist.entropy().mean()
                # if self.acmodel.use_broadcasting and not self.acmodel.always_broadcast:
                if not self.acmodel.always_broadcast:
                    broadcast_entropy = broadcast_dist.entropy().mean()
                    if self.acmodel.use_teamgrid:
                        broadcast_log_probs = broadcast_dist.log_prob(
                            sbs[j].broadcast.view(-1, 1, 1).repeat(1, self.num_options, self.num_actions))[
                            range(sbs[j].broadcast.shape[0]), sbs[j].current_options, sbs[j].action.long()]
                    else:
                        broadcast_log_probs = broadcast_dist.log_prob(
                            sbs[j].broadcast.view(-1, 1, 1).repeat(1, self.num_options, self.num_actions[j]))[
                            range(sbs[j].broadcast.shape[0]), sbs[j].current_options, sbs[j].action.long()]
                    broadcast_loss = -(broadcast_log_probs * (sbs[j].value_swa - sbs[j].value_sw)).mean()
                    # broadcast_loss = -(broadcast_log_probs * (sbs[j].value_swa_b - sbs[j].value_sw_b)).mean()
                    # broadcast_loss = -(broadcast_log_probs * (sbs_coord.value_swa - sbs_coord.value_sw)).mean()

                act_log_probs = act_dist.log_prob(sbs[j].action.view(-1, 1).repeat(1, self.num_options))[
                    range(sbs[j].action.shape[0]), sbs[j].current_options]
                policy_loss = -(act_log_probs * (sbs[j].value_swa - sbs[
                    j].value_sw)).mean()  # the second term should be coordinator value

                #policy_loss = (act_mlp.view(-1,1,1)[sbs[j].action.long()].squeeze() * (sbs[j].value_swa - sbs[j].value_sw)).mean() #doc-ml




                term_prob = term_dist.probs[range(sbs[j].action.shape[0]), sbs[j].current_options]
                termination_loss = (term_prob * (sbs[j].advantage + self.termination_reg)).mean()
                #termination_loss = (term_prob * (sbs_coord.advantage + self.termination_reg)).mean()


                if self.acmodel.use_broadcasting and not self.acmodel.always_broadcast:
                    loss = policy_loss \
                           - self.entropy_coef * entropy \
                            + broadcast_loss \
                            - self.entropy_coef * broadcast_entropy \
                           + self.term_loss_coef * termination_loss
                else:
                    loss = policy_loss \
                           - self.entropy_coef * entropy \
                           + self.term_loss_coef * termination_loss


                # Update batch values
                update_entropy[j] += entropy.item()
                # if self.acmodel.use_broadcasting and not self.acmodel.always_broadcast:
                if not self.acmodel.always_broadcast:
                    update_broadcast_entropy[j] += broadcast_entropy.item()
                    update_broadcast_loss[j] += broadcast_loss.item()
                update_policy_loss[j] += policy_loss.item()

                update_actor_loss[j] += loss

                # Collect agent embedding

                embeddings[j] = embedding

                # Collect masked (coord) embedding


                # if self.acmodel.use_broadcasting and self.acmodel.always_broadcast:
                if self.acmodel.always_broadcast:
                    masked_embeddings[j] = embedding
                    estimated_embeddings[j] = embedding
                else:
                    masked_embeddings[j] = sbs[j].broadcast.unsqueeze(1) * embedding
                    #estimated_embeddings[j] = sbs[j].broadcast.unsqueeze(1) * embedding + (1. - sbs[j].broadcast.unsqueeze(1)) * sbs[j].embedding
                    #estimated_embeddings[j] = sbs[j].broadcast.unsqueeze(1) * embedding + (1. - sbs[j].broadcast.unsqueeze(1)) * sbs[j].estimated_embedding
                    estimated_embeddings[j] = sbs[j].estimated_embedding
                    #print('check1', estimated_embeddings[j] == masked_embeddings[j], 'check2', estimated_embeddings[j] == embedding)



            # Central-critic forward propagation
            option_idxs = [sbs[j].current_options for j in range(self.num_agents)]
            action_idxs = [sbs[j].action for j in range(self.num_agents)]

            if self.acmodel.use_broadcasting:
                broadcast_idxs = [sbs[j].broadcast for j in range(self.num_agents)]

            _, value_a_b, _ = self.acmodel.forward_central_critic(estimated_embeddings, option_idxs, action_idxs, broadcast_idxs, sbs_coord.memory)

            value_losses = 0
            for j in range(self.num_agents):
                value_losses = value_losses + (value_a_b - sbs[j].target).pow(2).mean()
            value_loss = value_losses / self.num_agents

            # Update batch values

            # update_value += coord_value_action.mean().item()
            update_value += value_a_b.mean().item()
            update_value_loss += value_loss.item()
            update_critic_loss += self.value_loss_coef * value_loss

        # Re-initialize gradient buffers

        self.optimizer.zero_grad()
        # self.optimizer_critic.zero_grad()

        # Actors back propagation

        for j in range(self.num_agents):
            if not self.acmodel.use_teamgrid:
                # reduce lr for agent 1 (movable)
                # print('j', j, 'self.lr', self.lr)
                if j != 0:
                    lr = 0.1 * self.lr
                    self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                                         alpha=self.rmsprop_alpha, eps=self.rmsprop_eps)
                    self.optimizer.zero_grad()
                    # print('j', j, 'self.lr', self.lr, 'lr', lr)
            update_entropy[j] /= self.recurrence  # recurrence_agents

            update_policy_loss[j] /= self.recurrence
            if not self.always_broadcast:
                update_broadcast_entropy[j] /= self.recurrence
                update_broadcast_loss[j] /= self.recurrence

            update_actor_loss[j] /= self.recurrence
            update_values[j] /= self.recurrence
            # update_values_b[j] /= self.recurrence

            update_actor_loss[j].backward(retain_graph=True)

            # TODO: do we need the following back prop?
            # update_agent_critic_loss[j].backward(retain_graph=True)
            # update_agent_critic_loss_b[j].backward(retain_graph=True)

        # Critic back propagation

        update_value /= self.recurrence  # recurrence_coord
        update_value_loss /= self.recurrence

        # update_critic_loss.backward(retain_graph=True)
        update_critic_loss /= self.recurrence
        update_critic_loss.backward()

        # Learning step

        for name, param in self.acmodel.named_parameters():
            # print('name', name) #'param_data', param.data, 'param_grad', param.grad)
            if param.grad is None:
                print('Grad_none', name)

        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()


        #print('no_sil', self.no_sil)
        if not self.no_sil:
            mean_advs, mean_br_advs, num_samples = self.sil_model.train_sil_teamgrid((coord_exps, exps, logs), self.acmodel)
            #sil_max_reward = self.sil_model.get_best_reward()

        # Log some values
        logs["entropy"] = update_entropy
        logs["broadcast_entropy"] = update_broadcast_entropy
        logs["value"] = update_value
        logs["policy_loss"] = update_policy_loss
        logs["broadcast_loss"] = update_broadcast_loss
        logs["value_loss"] = update_value_loss
        logs["grad_norm"] = update_grad_norm
        logs["options"] = option_idxs
        logs["actions"] = action_idxs
        logs["broadcasts"] = broadcast_idxs

        # print('doc_log_return', logs["return_per_episode"])

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