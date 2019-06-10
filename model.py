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
                 use_memory=True,
                 use_text=False,
                 num_agents=2,
                 num_options=3,
                 use_act_values=True,
                 use_term_fn=True,
                 use_central_critic=True,
                 use_broadcasting=True,
                 ):
        super().__init__()

        # Decide which components are enabled
        self.use_act_values = use_act_values
        self.use_text = use_text
        self.use_memory = use_memory
        self.num_agents = num_agents
        self.num_options = num_options
        self.use_term_fn = use_term_fn
        self.use_central_critic = use_central_critic
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
                nn.Linear(64, self.num_options*2) # Check this; I believe this should be self.num_options * self.num_actions
            )

        # Define termination functions and option policy (policy over option)
        if self.use_term_fn:
            self.term_fn = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_options)
            )

        # Define central_critic model (sees all agents's embeddings)
        if self.use_act_values:
            central_critic_input_size = self.num_agents * (self.embedding_size + self.num_options + self.num_actions)
            central_critic_output_size = 1
        else:
            raise NotImplemented

        # central_critic needs its own memory
        self.coordinator_rnn = nn.LSTMCell(central_critic_input_size, central_critic_input_size)

        # Define agent_critic model (sees one agent embedding)
        # defines dimensionality
        if self.use_act_values:
            agent_critic_input_size = self.embedding_size
            agent_critic_output_size = actor_output_size
        else:
            agent_critic_input_size = self.embedding_size
            agent_critic_output_size = self.num_options

        # Defines the central_critic
        self.central_critic = nn.Sequential(
            nn.Linear(central_critic_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, central_critic_output_size)
        )

        # Defines the agent_critic
        self.agent_critic = nn.Sequential(
            nn.Linear(agent_critic_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, agent_critic_output_size)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    # Forward path for agent_critics to learn intra-option policies and broadcasts   
    def forward_agent_critic(self, obs, agent_memory):
        embedding, new_agent_memory = self._embed_observation(obs, agent_memory)

        x = self.actor(embedding).view((-1, self.num_options, self.num_actions))
        act_dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.agent_critic(embedding)
        agent_values = x.view((-1, self.num_options, self.num_actions)) if self.use_act_values else x.view((-1, self.num_options))


        if self.use_term_fn:
            x = self.term_fn(embedding).view((-1, self.num_options))
            term_dist = Bernoulli(probs=torch.sigmoid(x))

        else:
            term_dist = None

        if self.use_broadcasting:
            # x = self.broadcast_net(embedding).view((-1, self.num_options))
            x = self.broadcast_net(embedding).view((-1, self.num_options, 2))
            agent_values_b = x.view((-1, self.num_options, 2)) if self.use_act_values else x.view((-1, self.num_options))
            #broadcast_dist = Bernoulli(probs=torch.sigmoid(x))
            broadcast_dist = Categorical(logits=F.log_softmax(x, dim=-1)) # we need softmax on values depending on broadcast penalty
        return act_dist, agent_values, agent_values_b, new_agent_memory, term_dist, broadcast_dist, embedding

    def forward_central_critic(self, masked_embeddings, option_idxs, action_idxs, coordinator_memory):

        option_onehots = []
        for option_idxs_j in option_idxs:
            option_onehots.append(utils.idx_to_onehot(option_idxs_j.long(), self.num_options))

        action_onehots = []
        for action_idxs_j in action_idxs:
            action_onehots.append(utils.idx_to_onehot(action_idxs_j.long(), self.num_actions))

        coordinator_embedding = torch.cat([*masked_embeddings, *option_onehots, *action_onehots], dim=1)
        if self.use_memory:
            hidden = (coordinator_memory[:, :self.semi_memory_size], coordinator_memory[:, self.semi_memory_size:])
            hidden = self.coordinator_rnn(coordinator_embedding, hidden)
            coordinator_embedding = hidden[0]
            coordinator_memory = torch.cat(hidden, dim=1)

        assert self.use_central_critic
        values = self.central_critic(coordinator_embedding)

        return values.squeeze(), coordinator_memory.squeeze()

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def _embed_observation(self, obs, agent_memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (agent_memory[:, :self.semi_memory_size], agent_memory[:, self.semi_memory_size:])
            hidden = self.agent_memory_rnn(x, hidden)
            embedding = hidden[0]
            agent_memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        return embedding, agent_memory

    def get_number_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
