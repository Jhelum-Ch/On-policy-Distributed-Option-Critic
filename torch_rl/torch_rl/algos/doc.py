import numpy as np
import torch
import torch.nn.functional as F
import copy

from torch_rl.algos.base import BaseAlgo

class DOCAlgo(BaseAlgo):
    """The class for the Distributed Option-Critic algorithm."""

    # def __init__(self, num_agents, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
    #              entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence_agents=1, recurrence_coord=2,
    #              rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, num_options=4,
    #              termination_loss_coef=0.5, termination_reg=0.01, reshape_reward=None, always_broadcast = True, broadcast_penalty=-0.01):
    #     num_frames_per_proc = num_frames_per_proc or 8
    #
    #     super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
    #                      value_loss_coef, max_grad_norm, recurrence_agents, recurrence_coord, preprocess_obss, reshape_reward, broadcast_penalty, always_broadcast,
    #                      num_options, termination_loss_coef, termination_reg)

    def __init__(self, num_agents, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence = 4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, num_options=3,
                 termination_loss_coef=0.5, termination_reg=0.01, reshape_reward=None, always_broadcast = False, broadcast_penalty=-0.01):

        num_frames_per_proc = num_frames_per_proc or 8

        # super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
        #                  value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, broadcast_penalty, always_broadcast,
        #                  termination_loss_coef, termination_reg)
        super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, broadcast_penalty,
         termination_reg, termination_loss_coef)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self):

        # Collect experiences


        coord_exps, exps, logs = self.collect_experiences()
        # exps, logs = self.collect_experiences()


        # Compute starting indexes

        inds = self._get_starting_indexes()


        # Initialize containers for actors info


        sbs_coord           = None
        sbs                 = [None for _ in range(self.num_agents)]
        embeddings          = [None for _ in range(self.num_agents)]
        masked_embeddings   = [None for _ in range(self.num_agents)]
        estimated_embeddings = [None for _ in range(self.num_agents)]

        #last_broadcasts     = [None for _ in range(self.num_agents)]
       # last_masked_embeddings = [None for _ in range(self.num_agents)]


        if self.acmodel.recurrent:
            memories = [exps[j].memory[inds] for j in range(self.num_agents)]
            #coordinator_memory = coord_exps.memory[inds]

        update_entropy      = [0 for _ in range(self.num_agents)]
        update_broadcast_entropy = [0 for _ in range(self.num_agents)]
        update_policy_loss  = [0 for _ in range(self.num_agents)]
        update_broadcast_loss = [0 for _ in range(self.num_agents)]
        update_actor_loss   = [0 for _ in range(self.num_agents)]
        #update_agent_critic_loss = [0 for _ in range(self.num_agents)]

        # Initialize scalars for central-critic info

        update_value        = 0
        update_values = [0 for _ in range(self.num_agents)]
        #update_values_b = [0 for _ in range(self.num_agents)]
        update_value_loss   = 0
        #update_values_loss = [0 for _ in range(self.num_agents)]
        #update_values_b_loss = [0 for _ in range(self.num_agents)]
        #update_agent_critic_loss = [0 for _ in range(self.num_agents)]
        #update_agent_critic_loss_b = [0 for _ in range(self.num_agents)]
        update_critic_loss  = 0

        # Feed experience to model with gradient-tracking on a single process

        for i in range(self.recurrence): #recurrence_agents
            if self.acmodel.use_central_critic:
                sbs_coord = coord_exps[inds + i]


            for j in range(self.num_agents):


                # Create a sub-batch of experience



                sbs[j] = exps[j][inds + i]

                # Actor forward propagation

               # print('i', i, 'inds + i', len(inds + i), 'sbs_em', sbs[j].embeddings.size())

                if not self.acmodel.always_broadcast:
                    act_dist, _, _, memory, term_dist, broadcast_dist, embedding = self.acmodel.forward_agent_critic(sbs[j].obs, \
                                                                                                        memories[j] * \
                                                                                                        sbs[j].mask)
                else:
                    act_dist, _, memory, term_dist, embedding = self.acmodel.forward_agent_critic(
                        sbs[j].obs, \
                        memories[j] * \
                        sbs[j].mask)

                #print('embed', embedding.size())
                # To find the last successfully broadcast embedding
                # idx_list = torch.zeros(self.num_procs, device=self.device)
                # agent_successfully_broadcast_embedding = torch.zeros(self.num_procs, self.acmodel.semi_memory_size,
                #                                                      device=self.device)

                # for p in range(self.num_procs):
                #     id_list = [i for i, e in
                #                enumerate(self.rollout_broadcast_masks[j][:-1].transpose(0, 1)[p].tolist()) if
                #                e == 1]
                #     if len(id_list) != 0:
                #         idx_list[p] = max(id_list)
                #         agent_successfully_broadcast_embedding[p] = \
                #             self.rollout_agent_embeddings[j][:-1].transpose(0, 1)[p, idx_list[p].long()]
                #     else:
                #         agent_successfully_broadcast_embedding[p] = self.rollout_agent_embeddings[j, 0, p]
                # print('size_n', sbs[j].embeddings.reshape(self.num_procs,-1, self.acmodel.semi_memory_size).size())
                # for p in range(self.num_procs):
                #     id_list = [i for i, e in
                #                enumerate(sbs[j].embeddings.reshape(self.num_procs,-1, self.acmodel.semi_memory_size)[p].tolist()) if
                #                e == 1]
                #     if len(id_list) != 0:
                #         idx_list[p] = max(id_list)
                #         agent_successfully_broadcast_embedding[p] = \
                #             sbs[j].embeddings.reshape(self.num_procs,-1, self.acmodel.semi_memory_size)[p, idx_list[p].long()]
                #     else:
                #         agent_successfully_broadcast_embedding[p] = sbs[j].embeddings.reshape(self.num_procs,-1, self.acmodel.semi_memory_size)[p,0]


                #print('successful', agent_successfully_broadcast_embedding.size())

                # Actor losses

                entropy = act_dist.entropy().mean()
                # if self.acmodel.use_broadcasting and not self.acmodel.always_broadcast:
                if not self.acmodel.always_broadcast:
                    broadcast_entropy = broadcast_dist.entropy().mean()
                    broadcast_log_probs = broadcast_dist.log_prob(
                        sbs[j].broadcast.view(-1, 1, 1).repeat(1, self.num_options, self.num_actions))[
                        range(sbs[j].broadcast.shape[0]), sbs[j].current_options, sbs[j].action.long()]
                    broadcast_loss = -(broadcast_log_probs * (sbs[j].value_swa - sbs[j].value_sw)).mean()
                    # broadcast_loss = -(broadcast_log_probs * (sbs[j].value_swa_b - sbs[j].value_sw_b)).mean()
                    # broadcast_loss = -(broadcast_log_probs * (sbs_coord.value_swa - sbs_coord.value_sw)).mean()

                act_log_probs = act_dist.log_prob(sbs[j].action.view(-1, 1).repeat(1, self.num_options))[range(sbs[j].action.shape[0]), sbs[j].current_options]
                policy_loss = -(act_log_probs * (sbs[j].value_swa - sbs[j].value_sw)).mean() #the second term should be coordinator value
                #policy_loss = -(act_log_probs * (sbs_coord.value_swa - sbs_coord.value_sw)).mean()


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


                #curr_broadcast = broadcast_dist.sample() #[range(self.num_procs), self.current_options[j].long()]







                # # print(values[self.num_procs,sbs[j].current_options, sbs[j].action] == sbs[j].value_swa)
                # # Instead of values[self.num_procs,sbs[j].current_options, sbs[j].action] should we use sbs[j].value_swa
                # values_loss = (values[self.num_procs, sbs[j].current_options, sbs[j].action] - sbs[j].target).pow(
                #     2).mean()
                #
                # # Instead of values_b[self.num_procs,sbs[j].current_options, sbs[j].action] should we use sbs[j].value_swa_b?
                # values_b_loss = (
                #             values_b[self.num_procs, sbs[j].current_options, sbs[j].broadcast] - sbs[j].target_b).pow(
                #     2).mean()


            #print('masked', masked_embeddings[0].size(),'est', estimated_embeddings[0].size())


            # Central-critic forward propagation
            option_idxs = [sbs[j].current_options for j in range(self.num_agents)]
            action_idxs = [sbs[j].action for j in range(self.num_agents)]

            if self.acmodel.use_broadcasting:
                broadcast_idxs = [sbs[j].broadcast for j in range(self.num_agents)]

            _, value_a_b, _ = self.acmodel.forward_central_critic(estimated_embeddings, option_idxs, action_idxs, broadcast_idxs, sbs_coord.memory)

            # for j in range(self.num_agents):
            #     modified_masked_embeddings = copy.deepcopy(last_masked_embeddings)
            #     modified_masked_embeddings[j] = embeddings[j]
            #
            #
            #     values, _ = self.acmodel.forward_central_critic(modified_masked_embeddings, option_idxs, action_idxs, broadcast_idxs, sbs_coord.memory)
            #
            #     # TODO: debug L175 - L186
            #     value_a = torch.tensor(np.array(values)[:,0])
            #     values_loss = (value_a - sbs[j].target).pow(2).mean()
            #
            #
            #     values_b = torch.tensor(np.array(values)[:,1])
            #     values_b_loss = (values_b - sbs[j].target_b).pow(2).mean()
            #
            #     update_agent_critic_loss[j] += self.value_loss_coef * values_loss
            #     update_agent_critic_loss_b[j] += self.value_loss_coef * values_b_loss
            #
            #     update_values_loss[j] += values_loss.item()
            #     update_values_b_loss[j] += values_b_loss.item()
            #
            #     # Update agents values and values_b
            #     update_values[j] += sbs[j].value_swa.mean().item()
            #     update_values_b[j] += sbs[j].value_swa_b.mean().item()

            # Critic loss: this should use coord_value and coord_target

            # coord_value_action = torch.tensor(np.array(value_a_b)[:,0])
            # value_loss = (coord_value_action - sbs_coord.target).pow(2).mean()



            value_losses = 0
            for j in range(self.num_agents):
                value_losses = value_losses + (value_a_b - sbs[j].target).pow(2).mean()
            value_loss = value_losses / self.num_agents

            # Update batch values

            # update_value += coord_value_action.mean().item()
            update_value += value_a_b.mean().item()
            update_value_loss += value_loss.item()
            update_critic_loss += self.value_loss_coef * value_loss


        # for i in range(self.recurrence_coord):
        #     sbs_coord = coord_exps[inds + i]
        #
        #     option_idxs = [sbs[j].current_options for j in range(self.num_agents)]
        #     action_idxs = [sbs[j].action for j in range(self.num_agents)]
        #     broadcast_idxs = [sbs[j].broadcast for j in range(self.num_agents)]
        #
        #     value_a_b, _ = self.acmodel.forward_central_critic(masked_embeddings, option_idxs, action_idxs,
        #                                                        broadcast_idxs,
        #                                                        sbs_coord.memory)
        #
        #     value_losses = 0
        #     for j in range(self.num_agents):
        #         value_losses = value_losses + (value_a_b - sbs[j].target).pow(2).mean()
        #     value_loss = value_losses / self.num_agents
        #
        #     # Update batch values
        #
        #     # update_value += coord_value_action.mean().item()
        #     update_value += value_a_b.mean().item()
        #     update_value_loss += value_loss.item()
        #     update_critic_loss += self.value_loss_coef * value_loss



        # Re-initialize gradient buffers

        self.optimizer.zero_grad()

        # Actors back propagation

        for j in range(self.num_agents):

            update_entropy[j] /= self.recurrence #recurrence_agents

            update_policy_loss[j] /= self.recurrence
            if not self.always_broadcast:
                update_broadcast_entropy[j] /= self.recurrence
                update_broadcast_loss[j] /= self.recurrence

            update_actor_loss[j] /= self.recurrence
            update_values[j] /= self.recurrence
            #update_values_b[j] /= self.recurrence

            update_actor_loss[j].backward(retain_graph=True)

            # TODO: do we need the following back prop?
            # update_agent_critic_loss[j].backward(retain_graph=True)
            # update_agent_critic_loss_b[j].backward(retain_graph=True)




        # Critic back propagation

        update_value /= self.recurrence #recurrence_coord
        update_value_loss /= self.recurrence


        # update_critic_loss.backward(retain_graph=True)
        update_critic_loss.backward()


        # Learning step

        for name, param in self.acmodel.named_parameters():
            #print('name', name) #'param_data', param.data, 'param_grad', param.grad)
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
        logs["options"] = option_idxs
        logs["actions"] = action_idxs

        #print('doc_log_retun', logs["return_per_episode_with_broadcast_penalties"])

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
