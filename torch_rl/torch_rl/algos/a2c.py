import numpy
import torch
import torch.nn.functional as F

from torch_rl.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The class for the Advantage Actor-Critic algorithm."""

    def __init__(self, num_agents, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, num_options=2, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        # super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
        #                  value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, broadcast_penalty)
        super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

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
        masked_embeddings = [None for _ in range(self.num_agents)]
        estimated_embeddings = [None for _ in range(self.num_agents)]

        if self.acmodel.use_central_critic:
            sbs_coord = None
            update_critic_loss = 0
            update_value = 0
            update_value_loss = 0
            option_idxs = [None for _ in range(self.num_agents)]
            action_idxs = [None for _ in range(self.num_agents)]
            broadcast_idxs = [None for _ in range(self.num_agents)]
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

                # Initialize update values

                # update_entropy = 0
                # update_value = 0
                # update_policy_loss = 0
                # update_value_loss = 0
                # update_actor_loss = 0
                # update_critic_loss = 0


                # Initialize memory

                if self.acmodel.recurrent:
                    #memory = exps[j].memory[inds]
                    memory = memories[j]

                #for i in range(self.recurrence):

                    # Create a sub-batch of experience


                    sbs[j] = exps[j][inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        if not self.acmodel.always_broadcast:
                        # act_dist, values, memory, term_dist, _ = self.acmodel(sb.obs, memory * sb.mask)
                            act_dist, act_values, act_values_b, memory, _, broadcast_dist, embedding = self.acmodel.forward_agent_critic(sbs[j].obs, memory * sbs[j].mask)
                        else:
                            act_dist, act_values, memory, _, embedding = self.acmodel.forward_agent_critic(sbs[j].obs, memory * sbs[j].mask)
                    else:
                        if not self.acmodel.always_broadcast:
                            #act_dist, values = self.acmodel(sb.obs)
                            act_dist, act_values, act_values_b, _, _, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                                sbs[j].obs, memory * sbs[j].mask)
                        else:
                            act_dist, act_values, _, _, embedding = self.acmodel.forward_agent_critic(sbs[j].obs)

                    entropy = act_dist.entropy().mean()

                    agent_act_log_probs = act_dist.log_prob(sbs[j].action.view(-1, 1).repeat(1, self.num_options))[range(sbs[j].action.shape[0]), sbs[j].current_options]
                    agent_values = act_values[range(sbs[j].action.shape[0]), sbs[j].current_options]

                    policy_loss = -(agent_act_log_probs * sbs[j].advantage).mean()

                    if not self.acmodel.always_broadcast:
                        broadcast_entropy = broadcast_dist.entropy().mean()
                        broadcast_log_probs = broadcast_dist.log_prob(
                            sbs[j].broadcast.view(-1, 1, 1).repeat(1, self.num_options, self.num_actions))[
                            range(sbs[j].broadcast.shape[0]), sbs[j].current_options, sbs[j].action.long()]
                        broadcast_loss = -(broadcast_log_probs * sbs[j].advantage).mean()

                        loss = policy_loss - self.entropy_coef * entropy \
                               + broadcast_loss \
                               - self.entropy_coef * broadcast_entropy
                    else:
                        loss = policy_loss - self.entropy_coef * entropy \



                    # Update batch values

                    update_entropy[j] += entropy.item()
                    if not self.acmodel.always_broadcast:
                        update_broadcast_entropy[j] += broadcast_entropy.item()
                        update_broadcast_loss[j] += broadcast_loss.item()

                    update_policy_loss[j] += policy_loss.item()
                    update_actor_loss[j] += loss



                    if not self.acmodel.use_central_critic:
                        value_loss = (agent_values - sbs[j].returnn).pow(2).mean()

                        update_critic_loss[j] += self.value_loss_coef * value_loss

                        update_value[j] += agent_values.mean().item()
                        update_value_loss[j] += value_loss.item()


                    if self.acmodel.use_central_critic:
                        # Collect agent embedding

                        embeddings[j] = embedding
                        option_idxs[j] = sbs[j].current_options
                        action_idxs[j] = sbs[j].action
                        broadcast_idxs[j] = sbs[j].broadcast

                        # Collect masked (coord) embedding

                        # if self.acmodel.use_broadcasting and self.acmodel.always_broadcast:

                        if self.acmodel.always_broadcast:
                            masked_embeddings[j] = embedding
                            estimated_embeddings[j] = embedding
                        else:
                            masked_embeddings[j] = sbs[j].broadcast.unsqueeze(1) * embedding
                            estimated_embeddings[j] = sbs[j].broadcast.unsqueeze(1) * embedding + (1. - sbs[j].broadcast.unsqueeze(1)) * sbs[j].embedding


            if self.acmodel.use_central_critic:
               # Central-critic forward propagation
               #  option_idxs =[sbs[j].current_options for j in range(self.num_agents)]
               #  action_idxs = [sbs[j].action for j in range(self.num_agents)]

                # if self.acmodel.use_broadcasting:
                #     broadcast_idxs = [sbs[j].broadcast for j in range(self.num_agents)]

                _, value_a_b, _ = self.acmodel.forward_central_critic(estimated_embeddings, option_idxs,
                                                                   action_idxs, broadcast_idxs, sbs_coord.memory)

                avg_value_loss = 0
                for j in range(self.num_agents):
                    avg_value_loss = avg_value_loss + (value_a_b - sbs[j].returnn).pow(2).mean()
                value_loss = avg_value_loss / self.num_agents


               # update_value += coord_value_action.mean().item()
                update_value += value_a_b.mean().item()
                update_value_loss += value_loss.item()
                update_critic_loss += self.value_loss_coef * value_loss

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



           # Update update values

            # update_entropy /= self.recurrence
            # update_value /= self.recurrence
            # update_policy_loss /= self.recurrence
            # update_value_loss /= self.recurrence
            # update_loss /= self.recurrence

            # Update actor-critic

            #update_loss.backward()

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
            #print('name', name, 'param.data', param.data, 'param_grad', param.grad)
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

        print('doc_log_retun', logs["return_per_episode_with_broadcast_penalties"])

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

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
