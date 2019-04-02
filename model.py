import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch_rl
import gym
import utils

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, num_options=1):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.num_options = num_options
        self.curr_opt = 0

        if isinstance(action_space, gym.spaces.Discrete):
            self.num_actions = action_space.n
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model(s)
        self.actors = [
            nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_actions)
            )
            for _ in range(self.num_options)
        ]

        # Define critic's model
        if self.num_options > 1:
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size + self.num_options, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_actions)
            )

        else:
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Define termination functions and option policy (policy over option)
        if self.num_options > 1:
            self.opt_pol = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_options)
            )

            self.term_fns = [
                nn.Sequential(
                    nn.Linear(self.embedding_size, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
                for _ in range(self.num_options)
            ]

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, opt=None):
        if opt is None:
            opt = self.curr_opt

        embedding, new_memory =self.embed_observation(obs, memory)

        x = self.actors[opt](embedding)
        act_dist = Categorical(logits=F.log_softmax(x, dim=1))

        action = act_dist.sample()

        if self.num_options > 1:
            onehot_option = utils.idx_to_onehot(opt, self.num_options)
            # state-option q-values
            x = self.critic(torch.cat((embedding, onehot_option), dim=1))
        else:
            # state value
            x = self.critic(embedding)

        value = x.squeeze(1)

        if self.num_options > 1:
            x = self.term_fns[opt](embedding)
            term_dist = Bernoulli(probs=F.sigmoid(x))

            return action, act_dist, value, new_memory, term_dist

        else:

            return action, act_dist, value, new_memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def embed_observation(self, obs, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        return embedding, memory

    def choose_option(self, opt_values):
        self.curr_opt = torch.argmax(opt_values, dim=1)
