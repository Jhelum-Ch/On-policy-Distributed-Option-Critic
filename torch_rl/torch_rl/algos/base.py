from abc import ABC, abstractmethod
import torch
import numpy

from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 num_options=1, term_loss_coef=None, term_reg=None):
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
        term_loss_coef : float
            the weight of the termination loss in the final objective
        term_reg : float
            a small constant added to the option advantage to promote
            longer use of options (stretching)
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
        self.term_loss_coef = term_loss_coef
        self.term_reg = term_reg

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        if self.num_options > 1:
            self.current_options = torch.zeros(*shape, device=self.device, dtype=torch.int)
            self.option = torch.zeros(*shape, device=self.device, dtype=torch.int)
            self.terminates = torch.zeros(*shape, device=self.device, dtype=torch.int)
            self.state_opt_values = torch.zeros((*shape, self.num_options), device=self.device)
            self.selected_act_value = torch.zeros(*shape, device=self.device)

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

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.num_options > 1:
                    state_opt_value = torch.zeros[self.num_options, self.num_procs]
                    for opt in range(self.num_options):
                        if opt == self.acmodel.curr_opt:
                            action, act_dist, act_values, memory, term_dist = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), opt)
                            act_value = act_values[torch.arange(action.shape[0]), action]
                            state_opt_value[opt] = torch.sum(act_dist_tmp.prob() * act_values_tmp, dim=1)
                        else:
                            with torch.no_grad():
                                _, act_dist_tmp, act_values_tmp, _, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), opt)
                                state_opt_value[opt] = torch.sum(act_dist_tmp.prob() * act_values_tmp, dim=1)

                    state_value = torch.mean(self.state_opt_values, dim=2)[i]
                    terminate = term_dist.sample()

                    for t in terminate:
                        if t:


                else:
                    action, act_dist, state_value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            if self.num_options > 1:
                for terminate_per_proc in terminate:
                    if terminate_per_proc:
                        self.acmodel.curr_opt = torch.argmax(, dim=1)

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            if self.num_options > 1:
                self.selected_act_value[i] = act_value      # Q_U     (s,w,a)
                self.state_opt_values[i] = state_opt_value  # Q_omega (s,w)
                self.values[i] = state_value                # V_omega (s)
            else:
                self.values[i] = state_value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = act_dist.log_prob(action)
            if self.num_options > 1:
                self.terminates_prob[i] = term_dist.prob()
                self.terminates[i] = terminate

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

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.num_options > 1:
                # TODO: figure out where we need gradient for next_state computations
                next_action, next_act_dist, next_act_values, _, next_term_dist = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), opt)

            else:
                _, _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))

        for i in reversed(range(self.num_frames_per_proc)):
            if self.num_options > 1:
                self.advantages[i] = delta

            else:
                next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
                next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

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
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

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
