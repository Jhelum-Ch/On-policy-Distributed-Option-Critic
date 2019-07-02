import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch_rl
import gym
import utils
import copy
import numpy as np
#from multiagent.multi_discrete import MultiDiscrete


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
                 use_memory_agents=True,
                 use_memory_coord = True,
                 use_text=False,
                 num_agents=2,
                 num_options=3,
                 use_act_values=True,
                 use_term_fn=True,
                 use_central_critic=True,
                 use_broadcasting=True
                 ):
        super().__init__()

        # Decide which components are enabled
        self.use_act_values = use_act_values
        self.use_text = use_text
        self.use_memory_agents = use_memory_agents
        self.use_memory_coord = use_memory_coord
        self.num_agents = num_agents
        self.num_options = num_options
        self.use_term_fn = use_term_fn
        self.use_central_critic = use_central_critic
        self.use_broadcasting = use_broadcasting

        if isinstance(action_space, gym.spaces.Discrete):
            self.num_actions = action_space.n
        elif isinstance(action_space, MultiDiscrete):
            pass
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
        if self.use_memory_agents:
            self.agent_memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

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

        # Define actor's model(s): can we add broadcast head in the same model along with intra-option policies?
        actor_output_size = self.num_options * self.num_actions
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, actor_output_size)
        )

        # Define broadcast_net
        if self.use_broadcasting:
            self.broadcast_net = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_options * 2)
            )

        # Define termination functions and option policy (policy over option)
        if self.use_term_fn:
            self.term_fn = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_options)
            )

        # Define central_critic model (sees all agents's embeddings)
        central_critic_input_size = self.num_agents * (self.embedding_size + self.num_options + self.num_actions + 2) #2 for broadcast actions
        central_critic_output_size = 1


        # central_critic needs its own memory
        if self.use_memory_coord:
            self.coordinator_rnn = nn.LSTMCell(central_critic_input_size, central_critic_input_size)

        # Defines the central_critic
        self.central_critic = nn.Sequential(
            nn.Linear(central_critic_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, central_critic_output_size)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    @property
    def coord_memory_size(self):
        return 2*self.coord_semi_memory_size

    @property
    def coord_semi_memory_size(self):
        return self.num_agents * (self.embedding_size + self.num_options + self.num_actions + 2)

    # Forward path for agent_critics to learn intra-option policies and broadcasts   
    def forward_agent_critic(self, obs, agent_memory):
        embedding, new_agent_memory = self._embed_observation(obs, agent_memory)

        x = self.actor(embedding).view((-1, self.num_options, self.num_actions))
        act_dist = Categorical(logits=F.log_softmax(x, dim=-1))

        agent_values = x.view((-1, self.num_options, self.num_actions)) if self.use_act_values else x.view((-1, self.num_options))

        # x = self.agent_critic(embedding)
        # agent_values = x.view((-1, self.num_options, self.num_actions)) if self.use_act_values else x.view((-1, self.num_options))


        if self.use_term_fn:
            x = self.term_fn(embedding).view((-1, self.num_options))
            term_dist = Bernoulli(probs=torch.sigmoid(x))

        else:
            term_dist = None

        if self.use_broadcasting:
            x_b = self.broadcast_net(embedding).view((-1, self.num_options, 2))
            broadcast_dist = Categorical(
                logits=F.log_softmax(x_b, dim=-1))  # we need softmax on values depending on broadcast penalty

            agent_values_b = x_b.view((-1, self.num_options, 2)) if self.use_act_values else x_b.view((-1, self.num_options))

        return act_dist, agent_values, agent_values_b, new_agent_memory, term_dist, broadcast_dist, embedding



    def forward_central_critic(self, masked_embeddings, option_idxs, action_idxs, broadcast_idxs, coordinator_memory):
        option_onehots = []
        for option_idxs_j in option_idxs:
            option_onehots.append(utils.idx_to_onehot(option_idxs_j.long(), self.num_options))

        action_onehots = []
        for action_idxs_j in action_idxs:
            action_onehots.append(utils.idx_to_onehot(action_idxs_j.long(), self.num_actions))

        broadcast_onehots = []
        for broadcast_idxs_j in broadcast_idxs:
            broadcast_onehots.append(utils.idx_to_onehot(broadcast_idxs_j.long(), 2))


        coordinator_embedding = torch.cat([*masked_embeddings, *option_onehots, *action_onehots, *broadcast_onehots], dim=1)

        if self.use_memory_coord:
            # hidden = (coordinator_memory[:, :self.semi_memory_size], coordinator_memory[:, self.semi_memory_size:])
            hidden = (coordinator_memory[:, :self.coord_semi_memory_size], coordinator_memory[:, self.coord_semi_memory_size:])
            hidden = self.coordinator_rnn(coordinator_embedding, hidden)
            coordinator_embedding = hidden[0]
            coordinator_memory = torch.cat(hidden, dim=1)

        assert self.use_central_critic
        value = self.central_critic(coordinator_embedding)
        # value_a = torch.tensor(np.array(values)[:,0])
        # value_b = torch.tensor(np.array(values)[:,1])

        return value.squeeze(), coordinator_memory.squeeze()

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def _embed_observation(self, obs, agent_memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory_agents:
            hidden = (agent_memory[:, :self.semi_memory_size], agent_memory[:, self.semi_memory_size:])
            hidden = self.agent_memory_rnn(x, hidden)
            embedding = hidden[0]
            agent_memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        return embedding, agent_memory

    # def _embed_observation_with_others_broadcast(self, obs, modified_agent_memory):
    #     x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
    #     x = self.image_conv(x)
    #     x = x.reshape(x.shape[0], -1)
    #
    #     if self.use_memory:
    #         hidden = (modified_agent_memory[:, :self.semi_memory_size], modified_agent_memory[:, self.semi_memory_size:])
    #         hidden = self.agent_memory_rnn(x, hidden)
    #         embedding = hidden[0]
    #         modified_agent_memory = torch.cat(hidden, dim=1)
    #     else:
    #         embedding = x
    #
    #     if self.use_text:
    #         embed_text = self._get_embed_text(obs.text)
    #         embedding = torch.cat((embedding, embed_text), dim=1)
    #
    #     return embedding, modified_agent_memory

    def get_number_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
