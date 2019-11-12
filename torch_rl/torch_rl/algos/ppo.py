import numpy
import torch
import torch.nn.functional as F

from torch_rl.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""
    print('test')


    def __init__(self, num_agents=None, envs=None, acmodel=None, replay_buffer=None, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None, reshape_reward=None):

        num_frames_per_proc = num_frames_per_proc or 128

        # super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
        #                  value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, num_options)

        super().__init__(num_agents=num_agents, envs=envs, acmodel=acmodel, replay_buffer=replay_buffer, \
                         num_frames_per_proc=num_frames_per_proc, discount=discount, lr=lr, gae_lambda=gae_lambda,
                     entropy_coef=entropy_coef,
                     value_loss_coef=value_loss_coef, max_grad_norm=max_grad_norm, recurrence=recurrence, \
                         preprocess_obss=preprocess_obss, reshape_reward=reshape_reward
                     )

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        #assert self.batch_size % self.recurrence == 0

        if not self.acmodel.use_teamgrid and not self.acmodel.use_central_critic:
            a = self.acmodel.parametersList
            self.optimizer = torch.optim.Adam(a, lr, eps=adam_eps)
        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        #a = self.acmodel.parametersList
        #self.optimizer = torch.optim.Adam(a, lr, eps=adam_eps)
        # self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        # Compute starting indexes


        for j in range(self.num_agents):

            for _ in range(self.epochs):
                # Initialize log values

                log_entropies = []
                log_values = []
                log_policy_losses = []
                log_value_losses = []
                log_grad_norms = []

                for inds in self._get_batches_starting_indexes():
                    # Initialize batch values

                    batch_entropy = 0
                    batch_value = 0
                    batch_policy_loss = 0
                    batch_value_loss = 0
                    batch_loss = 0

                    # Initialize memory

                    if self.acmodel.recurrent:
                        memory = exps[j].memory[inds]

                    for i in range(self.recurrence):
                        # Create a sub-batch of experience

                        sb = exps[j][inds + i]

                        # Compute loss

                        # if self.acmodel.recurrent:
                        #     act_dist, values, memory, term_dist, _ = self.acmodel.forward_agent_critic(sb.obs, memory * sb.mask)
                        # else:
                        #     act_dist, values = self.acmodel.forward_agent_critic(sb.obs)

                        if self.acmodel.recurrent:
                            if not self.acmodel.always_broadcast:
                                # act_dist, values, memory, term_dist, _ = self.acmodel(sb.obs, memory * sb.mask)
                                act_mlp, act_dist, act_values, act_values_b, memory, _, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                                    sb.obs, memory * sb.mask, agent_index=j)
                            else:
                                act_mlp, act_dist, act_values, memory, _, embedding = self.acmodel.forward_agent_critic(
                                    sb.obs, memory * sb.mask, agent_index=j)
                        else:
                            if not self.acmodel.always_broadcast:
                                # act_dist, values = self.acmodel(sb.obs)
                                act_mlp, act_dist, act_values, act_values_b, _, _, broadcast_dist, embedding = self.acmodel.forward_agent_critic(
                                    sb.obs, memory * sb.mask, agent_index=j)
                            else:
                                act_mlp, act_dist, act_values, _, _, embedding = self.acmodel.forward_agent_critic(sb.obs, agent_index=j)

                        entropy = act_dist.entropy().mean()

                        agent_act_log_probs = act_dist.log_prob(sb.action.view(-1, 1).repeat(1, self.num_options))[range(sb.action.shape[0]), sb.current_options]
                        agent_values = act_values[range(sb.action.shape[0]), sb.current_options]

                        ratio = torch.exp(agent_act_log_probs - sb.log_prob)
                        surr1 = ratio * sb.advantage
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_clipped = sb.value + torch.clamp(agent_values - sb.value, -self.clip_eps, self.clip_eps)
                        surr1 = (agent_values - sb.returnn).pow(2)
                        surr2 = (value_clipped - sb.returnn).pow(2)
                        value_loss = torch.max(surr1, surr2).mean()

                        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                        # Update batch values

                        batch_entropy += entropy.item()
                        batch_value += agent_values.mean().item()
                        batch_policy_loss += policy_loss.item()
                        batch_value_loss += value_loss.item()
                        batch_loss += loss

                        # Update memories for next epoch

                        if self.acmodel.recurrent and i < self.recurrence - 1:
                            # exps.memory[inds + i + 1] = memory.detach()
                            exps[j].memory[inds + i + 1] = memory.detach()

                    # Update batch values

                    batch_entropy /= self.recurrence
                    batch_value /= self.recurrence
                    batch_policy_loss /= self.recurrence
                    batch_value_loss /= self.recurrence
                    batch_loss /= self.recurrence

                    # Update actor-critic

                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # Update log values

                    log_entropies.append(batch_entropy)
                    log_values.append(batch_value)
                    log_policy_losses.append(batch_policy_loss)
                    log_value_losses.append(batch_value_loss)
                    log_grad_norms.append(grad_norm)

            # Log some values

            logs["entropy"].append(numpy.mean(log_entropies))
            logs["value"].append(numpy.mean(log_values))
            logs["policy_loss"].append(numpy.mean(log_policy_losses))
            logs["value_loss"].append(numpy.mean(log_value_losses))
            logs["grad_norm"].append(numpy.mean(log_grad_norms))

        #print('ppo_log_return', logs["return_per_episode_with_broadcast_penalties"])

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
