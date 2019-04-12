import numpy
import torch
import torch.nn.functional as F

from torch_rl.algos.base import BaseAlgo

class OCAlgo(BaseAlgo):
    """The class for the Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, num_options=4,
                 termination_loss_coef=0.5, termination_reg=0.01, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         num_options, termination_loss_coef, termination_reg)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Forward propagation

            if self.acmodel.recurrent:
                act_dist, act_values, memory, term_dist = self.acmodel(sb.obs, memory * sb.mask)
            else:
                act_dist, act_values, _, term_dist = self.acmodel(sb.obs)

            # Compute losses

            entropy = act_dist.entropy().mean()

            act_log_probs = act_dist.log_prob(sb.action.view(-1, 1).repeat(1, self.num_options))[range(sb.action.shape[0]), sb.current_options]
            policy_loss = -(act_log_probs * (sb.value_swa - sb.value_sw)).mean()

            Q_U_swa = act_values[range(sb.action.shape[0]), sb.current_options, sb.action.long()]
            value_loss = (Q_U_swa - sb.delta).pow(2).mean()

            term_prob = term_dist.probs[range(sb.action.shape[0]), sb.current_options]
            termination_loss = - (term_prob * (sb.advantage + self.termination_reg)).mean()

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
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs["entropy"] = update_entropy
        logs["value"] = update_value
        logs["policy_loss"] = update_policy_loss
        logs["value_loss"] = update_value_loss
        logs["grad_norm"] = update_grad_norm

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
