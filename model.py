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
    def __init__(self,
                 obs_space,
                 action_space,
                 use_memory=False,
                 use_text=False,
                 num_agents=1,
                 num_options=1,
                 use_act_values=False,
                 use_term_fn=False,
                 use_centralized_critics=False,
                 use_broadcasting=False,
                 ):
        super().__init__()

        # Decide which components are enabled
        self.use_act_values = use_act_values
        self.use_text = use_text
        self.use_memory = use_memory
        self.num_agents = num_agents
        self.num_options = num_options
        self.use_term_fn = use_term_fn
        self.use_centralized_critics = use_centralized_critics
        self.use_broadcasting = use_broadcasting

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
        actor_output_size = self.num_actions * self.num_options
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, actor_output_size)
        )

        # Define critic's model
        if self.use_act_values:
            critic_output_size = actor_output_size

        elif not self.use_act_values:
            critic_output_size = self.num_options

        else:
            raise NotImplemented

        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, critic_output_size)
        )

        # Define termination functions and option policy (policy over option)
        if self.use_term_fn:
            self.term_fn = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_options)
            )

        # Define broadcast_net
        if self.use_broadcasting:
            assert self.use_centralized_critics
            self.broadcast_net = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_agents)
            )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        embedding, new_memory =self._embed_observation(obs, memory)

        x = self.actor(embedding).view((-1, self.num_options, self.num_actions))
        act_dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.critic(embedding)
        values = x.view((-1, self.num_options, self.num_actions)) if self.use_act_values else x.view((-1, self.num_options))

        if self.use_term_fn:
            x = self.term_fn(embedding).view((-1, self.num_options))
            term_dist = Bernoulli(probs=torch.sigmoid(x))

        else:
            term_dist = None

        return act_dist, values, new_memory, term_dist, embedding

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def _embed_observation(self, obs, memory):
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

    def get_number_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
