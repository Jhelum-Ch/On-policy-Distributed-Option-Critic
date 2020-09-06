import numpy
import torch
import torch.nn.functional as F

from torch_rl.algos.base import BaseAlgo


class OCAlgo(BaseAlgo):
    """The class for the Selfish Option-Critic algorithm."""

    def __init__(self, config=None, num_agents=None, envs=None, acmodel=None, replay_buffer=None, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, num_options=3,
                 termination_loss_coef=0.5, termination_reg= 0.01, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        # super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
        #                  value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
        #                  num_options, termination_loss_coef, termination_reg)
        super().__init__(config=config, num_agents=num_agents, envs=envs, acmodel=acmodel, replay_buffer=replay_buffer, \
                         num_frames_per_proc=num_frames_per_proc, discount=discount, lr=lr, gae_lambda=gae_lambda, \
                         entropy_coef=entropy_coef,
                 value_loss_coef=value_loss_coef, max_grad_norm=max_grad_norm, recurrence=recurrence, preprocess_obss=preprocess_obss, \
                         reshape_reward=reshape_reward,
                 termination_reg=termination_reg, termination_loss_coef=termination_loss_coef)

        if not self.acmodel.use_teamgrid and not self.acmodel.use_central_critic:
            a = self.acmodel.parametersList
            self.optimizer = torch.optim.RMSprop(a, lr, alpha=rmsprop_alpha, eps=rmsprop_alpha)
        else:
            self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                                 alpha=rmsprop_alpha, eps=rmsprop_eps)
        # a = self.acmodel.parametersList
        # print('a', a)
        # self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
        #                                      alpha=rmsprop_alpha, eps=rmsprop_eps)
        # import ipdb; ipdb.set_trace()
        # self.optimizer = torch.optim.RMSprop(a, lr, alpha=rmsprop_alpha, eps=rmsprop_alpha)
        # self.optimizer = []
        # for j in range(self.num_agents):
        #     self.optimizer.append(torch.optim.RMSprop(a[j], lr,
        #                                      alpha=rmsprop_alpha, eps=rmsprop_eps))

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        # Compute starting indexes

        inds = self._get_starting_indexes()

        for j in range(self.num_agents):

            # Initialize update values

            update_entropy = 0
            update_value = 0
            update_policy_loss = 0
            update_value_loss = 0
            update_loss = 0

            # Initialize memory

            if self.acmodel.recurrent:
                memory = exps[j].memory[inds]

            for i in range(self.recurrence):
                # Create a sub-batch of experience

                sb = exps[j][inds + i]
                # sbs_coord = coord_exps[inds + i]

                # Forward propagation

                if self.acmodel.recurrent:
                    if not self.acmodel.always_broadcast:
                        act_mlp, act_dist, act_values, _, memory, term_dist, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                            sb.obs, memory * sb.mask, agent_index=j, sil_module=False)
                    else:
                        act_mlp, act_dist, act_values, memory, term_dist, embedding = self.acmodel.forward_agent_critic(
                            sb.obs, memory * sb.mask, agent_index=j, sil_module=False)

                else:
                    if not self.acmodel.always_broadcast:
                        act_mlp, act_dist, act_values, _, _, term_dist, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                            sb.obs, memory * sb.mask, agent_index=j, sil_module=False)
                    else:
                        act_mlp, act_dist, act_values, _, term_dist, embedding = self.acmodel.forward_agent_critic(
                            sb.obs, agent_index=j, sil_module=False)

                # Compute losses

                entropy = act_dist.entropy().mean()

                act_log_probs = act_dist.log_prob(sb.action.view(-1, 1).repeat(1, self.num_options))[
                    range(sb.action.shape[0]), sb.current_options]
                policy_loss = -(act_log_probs * (sb.value_swa - sb.value_sw)).mean()

                Q_U_swa = act_values[range(sb.action.shape[0]), sb.current_options, sb.action.long()]
                value_loss = (Q_U_swa - sb.target).pow(2).mean()

                term_prob = term_dist.probs[range(sb.action.shape[0]), sb.current_options]
                # print('sb.adv', sb.advantage, 'self.termination_reg', self.termination_reg)
                termination_loss = (term_prob * (sb.advantage + self.termination_reg)).mean()

                loss = policy_loss \
                       - self.entropy_coef * entropy \
                       + self.value_loss_coef * value_loss \
                       + self.term_loss_coef * termination_loss

                # Update batch values

                update_entropy += entropy.item()
                update_value += Q_U_swa.mean().item()
                update_policy_loss += policy_loss.item()
                update_value_loss += value_loss.item()
                update_loss += loss

            # Update update values

            update_entropy /= self.recurrence
            update_value /= self.recurrence
            update_policy_loss /= self.recurrence
            update_value_loss /= self.recurrence
            update_loss /= self.recurrence

            # Update actor-critic

            self.optimizer.zero_grad()
            update_loss.backward()

            for name, param in self.acmodel.named_parameters():
                # print('name', name) #'param_data', param.data, 'param_grad', param.grad)
                if param.grad is None:
                    print('Grad_none', name)
            update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in a[j]) ** 0.5
            # torch.nn.utils.clip_grad_norm_(a[j], self.max_grad_norm)
            # self.optimizer[j].step()

            # Log some values

            logs["entropy"].append(update_entropy)
            logs["value"].append(update_value)
            logs["policy_loss"].append(update_policy_loss)
            logs["value_loss"].append(update_value_loss)
            logs["grad_norm"].append(update_grad_norm)

            # print('oc_log_retun', logs["return_per_episode_with_broadcast_penalties"])

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
