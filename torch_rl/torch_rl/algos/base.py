from abc import ABC, abstractmethod
import torch
import numpy

from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 num_options=1, termination_loss_coef=None, termination_reg=None, option_epsilon=0.05):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        num_options : int
            the number of options
        termination_loss_coef : float
            the weight of the termination loss in the final objective
        termination_reg : float
            a small constant added to the option advantage to promote
            longer use of options (stretching)
        option_epsilon : float
            a small constant for the epsilon-soft policy over options
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.num_options = num_options
        self.termination_loss_coef = termination_loss_coef
        self.termination_reg = termination_reg
        self.option_epsilon = option_epsilon


        # Dimension convention

        if self.num_options > 1:
            self.batch_dim = 0
            self.opt_dim = 1
            self.act_dim = 2
        else:
            self.batch_dim = 0
            self.act_dim = 1
            self.opt_dim = None

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc + 1, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.done_mask = torch.ones(shape[1], device=self.device)
        self.done_masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        if self.num_options > 1:
            self.current_option = torch.randint(low=0, high=self.num_options, size=(self.num_procs,), device=self.device, dtype=torch.float)
            self.current_options = torch.zeros(*shape, device=self.device)

            self.terminates = torch.zeros(*shape, device=self.device)
            self.terminates_prob = torch.zeros(*shape, device=self.device)

            self.values_s_w_a = torch.zeros(*shape, device=self.device)
            self.values_s_w = torch.zeros(*shape, device=self.device)
            self.values_s = torch.zeros(*shape, device=self.device)

            self.deltas = torch.zeros(*shape, device=self.device)

        else:
            self.values = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        if not self.recurrence:
            raise Exception("Deprecated: self.recurrence has to be True."
                            "If no reccurence is used, we will still have self.recurrence=True"
                            "but self.acmodel.use_memory will be set to False.")

        rollout_length = len(self.obss)
        for i in range(rollout_length):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                if self.num_options > 1:
                    act_dist, act_values, memory, term_dist = self.acmodel(preprocessed_obs, self.memory * self.done_mask.unsqueeze(1))

                else:
                    act_dist, state_value, memory = self.acmodel(preprocessed_obs, self.memory * self.done_mask.unsqueeze(1))

            # select action
            action = act_dist.sample()
            if self.num_options > 1:
                action = action[range(self.num_procs), self.current_option.long()]

            # environment setp

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # compute OC-specific quantities

            if self.num_options > 1:
                Q_omega_sw = torch.sum(act_dist.probs * act_values, dim=self.act_dim, keepdim=True)

                terminate = term_dist.sample()[range(self.num_procs), self.current_option.long()]
                # change current_option w.r.t. termination
                self.current_option = (self.current_option * (1. - terminate)) + (terminate * torch.argmax(Q_omega_sw, dim=self.opt_dim).squeeze().float())

                Q_U_swa    = act_values[range(self.num_procs), self.current_option.long(), action]
                Q_omega_sw = Q_omega_sw[range(self.num_procs), self.current_option.long(), :]
                V_omega_s  = torch.mean(torch.sum(act_dist.probs * act_values, dim=self.act_dim, keepdim=True), dim=self.opt_dim, keepdim=True)

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.done_masks[i] = self.done_mask
            self.done_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action

            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)

            if self.num_options > 1:
                self.values_s_w_a[i] = Q_U_swa
                self.values_s_w[i]   = Q_omega_sw.squeeze()
                self.values_s[i]     = V_omega_s.squeeze()

                self.log_probs[i] = act_dist.logits[range(self.num_procs), self.current_option.long(), action]
                self.terminates_prob[i] = term_dist.probs[range(self.num_procs), self.current_option.long()]
                self.terminates[i] = terminate

                self.current_options[i] = self.current_option
                # change current_option w.r.t. episode ending
                self.current_option = (self.current_option * (1. - self.done_mask)) + (self.done_mask * torch.randint(low=0, high=self.num_options, size=(self.num_procs,), device=self.device, dtype=torch.float))

            else:
                self.values[i] = state_value
                self.log_probs[i] = act_dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.done_mask
            self.log_episode_reshaped_return *= self.done_mask
            self.log_episode_num_frames *= self.done_mask

        # Add advantage and return to experiences

        for i in reversed(range(self.num_frames_per_proc)):
            if self.num_options > 1:

                self.deltas[i] = self.rewards[i] - self.values_s_w_a[i] + \
                        (1. - self.done_masks[i+1]) * (1. - self.terminates[i+1]) * \
                        (
                                self.discount * (1. - self.terminates_prob[i+1]) * self.values_s_w[i+1] +
                                self.discount * self.terminates_prob[i+1] * torch.max(self.values_s_w[i+1])
                        )

                self.advantages[i] = self.values_s_w[i+1] - self.values_s[i+1]

            else:
                next_mask = self.done_masks[i + 1]
                next_value = self.values[i+1]
                next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc else 0

                delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
                self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories[:-1].transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.done_masks[:-1].transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions[:-1].transpose(0, 1).reshape(-1)
        exps.reward = self.rewards[:-1].transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages[:-1].transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs[:-1].transpose(0, 1).reshape(-1)
        if self.num_options > 1:
            exps.current_options = self.current_options[:-1].transpose(0, 1).reshape(-1)

            exps.terminate = self.terminates[:-1].transpose(0, 1).reshape(-1)
            exps.terminate_prob = self.terminates_prob[:-1].transpose(0, 1).reshape(-1)

            exps.value_s_w_a = self.values_s_w_a[:-1].transpose(0, 1).reshape(-1)
            exps.value_s_w = self.values_s_w[:-1].transpose(0, 1).reshape(-1)
            exps.value_s = self.values_s[:-1].transpose(0, 1).reshape(-1)

            exps.delta = self.deltas[:-1].transpose(0, 1).reshape(-1)

        else:
            exps.value = self.values[:-1].transpose(0, 1).reshape(-1)
            exps.returnn = exps[:-1].value + exps[:-1].advantage

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
