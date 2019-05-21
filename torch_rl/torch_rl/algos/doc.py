import numpy
import torch
import torch.nn.functional as F

from torch_rl.algos.base import BaseAlgo

class DOCAlgo(BaseAlgo):
    """The class for the Advantage Actor-Critic algorithm."""

    def __init__(self, num_agents, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, num_options=4,
                 termination_loss_coef=0.5, termination_reg=0.01, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(num_agents, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         num_options, termination_loss_coef, termination_reg)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self):

        # Collect experiences

        exps, logs = self.collect_experiences()

        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize containers for actors info

        sbs                 = [None for _ in range(self.num_agents)]
        embeddings          = [None for _ in range(self.num_agents)]
        if self.acmodel.recurrent:
            memories = [exps[j].memory[inds] for j in range(self.num_agents)]

        update_entropy      = [0 for _ in range(self.num_agents)]
        update_policy_loss  = [0 for _ in range(self.num_agents)]
        update_actor_loss   = [0 for _ in range(self.num_agents)]

        # Initialize scalars for central-critic info

        update_value        = 0
        update_value_loss   = 0
        update_critic_loss  = 0

        # Feed experience to model with gradient-tracking on a single process

        for i in range(self.recurrence):

            for j in range(self.num_agents):

                # Create a sub-batch of experience

                sbs[j] = exps[j][inds + i]

                # Actor forward propagation

                act_dist, _, memory, term_dist, embedding = self.acmodel(sbs[j].obs, memories[j] * sbs[j].mask)

                # Actor losses

                entropy = act_dist.entropy().mean()

                act_log_probs = act_dist.log_prob(sbs[j].action.view(-1, 1).repeat(1, self.num_options))[range(sbs[j].action.shape[0]), sbs[j].current_options]
                policy_loss = -(act_log_probs * (sbs[j].value_swa - sbs[j].value_sw)).mean()

                term_prob = term_dist.probs[range(sbs[j].action.shape[0]), sbs[j].current_options]
                termination_loss = (term_prob * (sbs[j].advantage + self.termination_reg)).mean()

                loss = policy_loss \
                       - self.entropy_coef * entropy \
                       + self.term_loss_coef * termination_loss

                # Update batch values

                update_entropy[j] += entropy.item()
                update_policy_loss[j] += policy_loss.item()
                update_actor_loss[j] += loss

                # Collect agent embedding

                embeddings[j] = embedding

            # Central-critic forward propagation

            option_idxs = [sbs[j].current_options for j in range(self.num_agents)]
            action_idxs = [sbs[j].action for j in range(self.num_agents)]

            values = self.acmodel.forward_central_critic(embeddings, option_idxs, action_idxs)

            # Critic loss

            value_losses = 0
            for j in range(self.num_agents):
                value_losses = value_losses + (values - sbs[j].target).pow(2).mean()
            value_loss = value_losses / self.num_agents

            # Update batch values

            update_value += values.mean().item()
            update_value_loss += value_loss.item()
            update_critic_loss += self.value_loss_coef * value_loss


        # Re-initialize gradient buffers

        self.optimizer.zero_grad()

        # Actors back propagation

        for j in range(self.num_agents):

            update_entropy[j] /= self.recurrence
            update_policy_loss[j] /= self.recurrence
            update_actor_loss[j] /= self.recurrence

            update_actor_loss[j].backward(retain_graph=True)

        # Critic back propagation

        update_value /= self.recurrence
        update_value_loss /= self.recurrence

        update_critic_loss.backward()

        # Learning step

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
