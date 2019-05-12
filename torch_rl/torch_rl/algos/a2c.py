import numpy
import torch
import torch.nn.functional as F

from torch_rl.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The class for the Advantage Actor-Critic algorithm."""

    def __init__(self, num_agents, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, num_options=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, num_options)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

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

                # Compute loss

                if self.acmodel.recurrent:
                    act_dist, values, memory, term_dist = self.acmodel(sb.obs, memory * sb.mask)
                else:
                    act_dist, values = self.acmodel(sb.obs)

                entropy = act_dist.entropy().mean()

                agent_act_log_probs = act_dist.log_prob(sb.action.view(-1, 1).repeat(1, self.num_options))[range(sb.action.shape[0]), sb.current_options]
                agent_values = values[range(sb.action.shape[0]), sb.current_options]

                policy_loss = -(agent_act_log_probs * sb.advantage).mean()

                value_loss = (agent_values - sb.returnn).pow(2).mean()

                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update batch values

                update_entropy += entropy.item()
                update_value += agent_values.mean().item()
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
            update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Log some values

            logs["entropy"].append(update_entropy)
            logs["value"].append(update_value)
            logs["policy_loss"].append(update_policy_loss)
            logs["value_loss"].append(update_value_loss)
            logs["grad_norm"].append(update_grad_norm)

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
