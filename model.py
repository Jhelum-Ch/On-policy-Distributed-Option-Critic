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
from multiagent.multi_discrete import MultiDiscrete


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
                 termination_reg,
                 use_teamgrid,
                 always_broadcast,
                 use_memory_agents = True,
                 use_memory_coord = True,
                 use_text=False,
                 num_agents=2,
                 num_options= 2, # TODO: choose num_options = 1 for selfish A2C and PPO;
                 use_act_values=True,
                 use_term_fn=True, # TODO: choose False for A2C and PPO
                 use_central_critic=False, # TODO: True for DOC, False for OC and PPO
                 use_broadcasting=True, #TODO: Always True
                 #always_broadcast = not use_central_critic
                 ):
        super().__init__()

        # Decide which components are enabled
        self.use_act_values = use_act_values
        self.use_text = use_text
        self.use_memory_agents = use_memory_agents
        self.use_memory_coord = use_memory_coord
        #self.scenario = scenario
        self.num_agents = num_agents
        self.num_options = num_options
        self.use_central_critic = use_central_critic
        self.always_broadcast = not self.use_central_critic if not self.use_central_critic else always_broadcast
        self.num_broadcasts = 1 if self.always_broadcast or not self.recurrent else 2 #1 if self.always_broadcast else 2
        self.use_term_fn = use_term_fn
        self.use_broadcasting = use_broadcasting
        self.termination_reg = termination_reg

        self.use_teamgrid = use_teamgrid
        #self.agent_index = agent_index


        if isinstance(action_space, gym.spaces.Discrete):
            #print('Hi1', action_space)
            self.num_actions = action_space.n
            self.agents_actions = self.num_actions
        elif isinstance(action_space, MultiDiscrete):
            pass
        elif isinstance(action_space, list):
            list_actions = [action_space[i].n for i in range(len(action_space))]
            if len(set(list_actions)) < len(list_actions): #repeated num_actions
                for i in range(len(action_space)):
                    self.num_actions = action_space[0].n
            else:
                self.num_actions = [action_space[i].n for i in range(len(action_space))]

            #print('self.num_actions', self.num_actions)
            #print('Hi3', action_space)
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        #print('obs_space', obs_space)

        if isinstance(obs_space["image"],list):
            n = [obs_space["image"][i][0] for i in range(len(obs_space["image"]))]
            #m = [obs_space["image"][i][1] for i in range(len(obs_space["image"]))]
            self.image_embedding_size = n
            #print('self.image_embedding_size', self.image_embedding_size)
        else:
            n = obs_space["image"][0]
            #print('obs_shape', obs_space["image"])
            m = obs_space["image"][1]
            self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define image embedding
        if self.use_teamgrid:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU()
            )
        # else:
        #     #print('image_embedding_size', self.image_embedding_size)
        #     self.image_conv = nn.Sequential(
        #         nn.Linear(8, 16),
        #         nn.ReLU(),
        #         nn.Linear(16, 64),
        #         nn.ReLU()
        #     )



        # Define memory
        if self.use_memory_agents:
            if isinstance(self.image_embedding_size, list):
                self.agent_memory_rnn = []
                for j in range(self.num_agents):
                    self.agent_memory_rnn.append(nn.LSTMCell(self.image_embedding_size[j], self.semi_memory_size[j]))
            else: # int
                self.agent_memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size #could be a list
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model(s): can we add broadcast head in the same model along with intra-option policies?
        if self.use_teamgrid:
            actor_output_size = self.num_options * self.num_actions
            #print('self_num_op', self.num_options, 'self.num_ac', self.num_actions,'actor_out', actor_output_size)
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, actor_output_size)
            )
        else:
            actor_output_size = []
            self.actor = []
            for j in range(self.num_agents):
                actor_output_size.append(self.num_options*self.num_actions[j])
                self.actor.append(nn.Sequential(
                nn.Linear(self.embedding_size[j], 64),
                nn.Tanh(),
                nn.Linear(64, actor_output_size[j])
            ))
            actor_params = []
            for j in range(self.num_agents):
                actor_params.extend(list(self.actor[j].parameters()))
            # actor_params = [list(self.actor[j].parameters()) for j in range(self.num_agents)]
            # actor_output_size = self.num_options * self.num_actions[agent_index]
            # # print('self_num_op', self.num_options, 'self.num_ac', self.num_actions,'actor_out', actor_output_size)
            # self.actor = nn.Sequential(
            #     nn.Linear(self.embedding_size[agent_index], 64),
            #     nn.Tanh(),
            #     nn.Linear(64, actor_output_size)
            # )


        # Define broadcast_net
        # if self.use_broadcasting and not self.always_broadcast:
        if not self.always_broadcast:
            if self.use_teamgrid:
                self.broadcast_net = nn.Sequential(
                    nn.Linear(self.embedding_size, 64),
                    nn.Tanh(),
                    nn.Linear(64, self.num_options * self.num_actions * self.num_broadcasts)
                )
            else:
                self.broadcast_net = []
                for j in range(self.num_agents):
                    self.broadcast_net.append(nn.Sequential(
                    nn.Linear(self.embedding_size[j], 64),
                    nn.Tanh(),
                    nn.Linear(64, self.num_options * self.num_actions[j] * self.num_broadcasts)
                ))
                # self.broadcast_net = nn.Sequential(
                #     nn.Linear(self.embedding_size[agent_index], 64),
                #     nn.Tanh(),
                #     nn.Linear(64, self.num_options * self.num_actions[agent_index] * self.num_broadcasts)
                # )
                broadcast_params = []
                for j in range(self.num_agents):
                    broadcast_params.extend(list(self.broadcast_net[j].parameters()))
                # broadcast_params = [list(self.broadcast_net[j].parameters()) for j in range(self.num_agents)]

        # Define termination functions and option policy (policy over option)
        if self.use_term_fn:
            if self.use_teamgrid:
                self.term_fn = nn.Sequential(
                    nn.Linear(self.embedding_size, 64),
                    nn.Tanh(),
                    nn.Linear(64, self.num_options)
                )
            else:
                self.term_fn = []
                for j in range(self.num_agents):
                    self.term_fn.append(nn.Sequential(
                    nn.Linear(self.embedding_size[j], 64),
                    nn.Tanh(),
                    nn.Linear(64, self.num_options)
                ))
                term_params = []
                for j in range(self.num_agents):
                    term_params.extend(list(self.term_fn[j].parameters()))
                # term_params = [list(self.term_fn[j].parameters()) for j in range(self.num_agents)]
                # self.term_fn = nn.Sequential(
                #     nn.Linear(self.embedding_size[agent_index], 64),
                #     nn.Tanh(),
                #     nn.Linear(64, self.num_options)
                # )

        if self.use_central_critic:
            # Define central_critic model (sees all agents's embeddings)
            if self.use_teamgrid:
                critic_input_size = self.num_agents * (
                        self.embedding_size + self.num_options + self.num_actions + self.num_broadcasts)
                critic_output_size = 1

                # central_critic needs its own memory
                if self.use_memory_coord:
                    self.coordinator_rnn = nn.LSTMCell(critic_input_size, critic_input_size)
            else:
                critic_input_size = int(np.sum(self.embedding_size)) + self.num_agents * (self.num_options + self.num_broadcasts) + int(np.sum(self.num_actions))
                # critic_input_size = int(np.sum(self.embedding_size)) + self.num_agents * (
                #             self.num_options + self.num_broadcasts + max(self.num_actions))
                #print('critic_input_size', critic_input_size)
                critic_output_size = 1

                # central_critic needs its own memory
                if self.use_memory_coord:
                    self.coordinator_rnn = nn.LSTMCell(critic_input_size, critic_input_size)

            # for param in self.target_critic.parameters():
            #     param.requires_grad = False
        else:
            # Define regular critic model (sees one agent embedding)
            # defines dimensionality
            if self.use_teamgrid:
                if self.use_act_values:
                    critic_input_size = self.embedding_size
                    critic_output_size = self.num_options * self.num_actions * self.num_broadcasts
                    #critic_output_size = actor_output_size
                else:
                    critic_input_size = self.embedding_size
                    critic_output_size = self.num_options
            else:
                if self.use_act_values:
                    critic_input_size = []
                    critic_output_size = []
                    for j in range(self.num_agents):
                        critic_input_size.append(self.embedding_size[j])
                        critic_output_size.append(self.num_options * self.num_actions[j] * self.num_broadcasts)
                    #critic_output_size = actor_output_size
                else:
                    critic_input_size = []
                    for j in range(self.num_agents):
                        critic_input_size.append(self.embedding_size[j])
                    critic_output_size = self.num_options
                # if self.use_act_values:
                #     critic_input_size = self.embedding_size[agent_index]
                #     critic_output_size = self.num_options * self.num_actions[agent_index] * self.num_broadcasts
                #     #critic_output_size = actor_output_size
                # else:
                #     critic_input_size = self.embedding_size[agent_index]
                #     critic_output_size = self.num_options

            #print('critic_input_size', critic_input_size, 'critic_output_size', critic_output_size)

        # Defines the critic
        # self.critic = nn.Sequential(
        #     nn.Linear(critic_input_size, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, critic_output_size)
        # )
        # import ipdb;
        # ipdb.set_trace()
        if isinstance(critic_input_size, int) and isinstance(critic_output_size, int):
            self.critic = nn.Sequential(
                nn.Linear(critic_input_size, 64),
                nn.Tanh(),
                nn.Linear(64, critic_output_size)
            )
        elif isinstance(critic_input_size, int) and isinstance(critic_output_size, list):
            self.critic = []
            for j in range(self.num_agents):
                self.critic.append(nn.Sequential(
                    nn.Linear(critic_input_size, 64),
                    nn.Tanh(),
                    nn.Linear(64, critic_output_size[j])
                ))
            critic_params = []
            for j in range(self.num_agents):
                critic_params.extend(list(self.critic[j].parameters()))

        elif isinstance(critic_input_size, list) and isinstance(critic_output_size, int): #list
            self.critic = []
            for j in range(self.num_agents):
                self.critic.append(nn.Sequential(
                    nn.Linear(critic_input_size[j], 64),
                    nn.Tanh(),
                    nn.Linear(64, critic_output_size)
                ))
            critic_params = []
            for j in range(self.num_agents):
                critic_params.extend(list(self.critic[j].parameters()))
        else:
            self.critic = []
            for j in range(self.num_agents):
                self.critic.append(nn.Sequential(
                    nn.Linear(critic_input_size[j], 64),
                    nn.Tanh(),
                    nn.Linear(64, critic_output_size[j])
                ))
            critic_params = []
            for j in range(self.num_agents):
                critic_params.extend(list(self.critic[j].parameters()))
            # critic_params = [list(self.critic[j].parameters()) for j in range(self.num_agents)]

        # self.parametersList = nn.ParameterList(actor_params + term_params + critic_params)
        #import ipdb; ipdb.set_trace()
        if not self.use_teamgrid and not self.use_central_critic:

            #print('parameter', torch.nn.Parameter(self.actor[j]))
            if self.always_broadcast:
                if self.num_options > 1:
                    # self.parametersList = [nn.ParameterList([actor_params[j], term_params[j],critic_params[j]]) for j in range(self.num_agents)]
                    # self.parametersList = [[actor_params[j], term_params[j], critic_params[j]] for j in
                    #                        range(self.num_agents)]
                    self.parametersList = actor_params + critic_params + term_params
                else:
                    self.parametersList = actor_params + critic_params
            else:
                if self.num_options > 1:
                    # self.parametersList = [nn.ParameterList([actor_params[j],term_params[j],critic_params[j],broadcast_params[j]]) for j in range(self.num_agents)]
                    # self.parametersList = [
                    #     [actor_params[j], term_params[j], critic_params[j], broadcast_params[j]] for j in
                    #     range(self.num_agents)]
                    self.parametersList = actor_params + critic_params + term_params + broadcast_params
                else:
                    self.parametersList = actor_params + critic_params + broadcast_params

        #print('LEN', len(self.parametersList))
        #print('len_paramList',len(self.parametersList), 'paramsList', self.parametersList)
        # # defines the target critic
        # self.target_critic = nn.Sequential(
        #     nn.Linear(critic_input_size, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, critic_output_size)
        # )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        if isinstance(self.semi_memory_size, int):
            return 2*self.semi_memory_size
        else:
            return [2*self.semi_memory_size[j] for j in range(self.num_agents)]

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    @property
    def coord_memory_size(self):
        return 2*self.coord_semi_memory_size

    @property
    def coord_semi_memory_size(self):
        if self.use_teamgrid:
            if self.num_options is not None:
                return self.num_agents * (self.embedding_size + self.num_options + self.num_actions + self.num_broadcasts)
            else:
                return self.num_agents * (self.embedding_size + self.num_actions + self.num_broadcasts)
        else:
            if self.num_options is not None:
                return  int(np.sum(self.embedding_size)) + self.num_agents * (self.num_options + self.num_broadcasts) + int(np.sum(self.num_actions))
                # return int(np.sum(self.embedding_size)) + self.num_agents * (
                #             self.num_options + self.num_broadcasts + max(self.num_actions))
            else:
                return  int(np.sum((self.embedding_size))) + self.num_agents * self.num_broadcasts + int(np.sum(self.num_actions))
                #return int(np.sum((self.embedding_size))) + self.num_agents * (self.num_broadcasts + max(self.num_actions))

    # Forward path for agent_critics to learn intra-option policies and broadcasts
    def forward_agent_critic(self, obs, agent_memory, agent_index):
        #print('obs', obs)
        embedding, new_agent_memory = self._embed_observation(obs, agent_memory, agent_index)
        #print('agent_index', agent_index)
        if self.use_teamgrid:
            x = self.actor(embedding).view((-1, self.num_options, self.num_actions))
        else:
            #print('Hi_j',agent_index)
            x = self.actor[agent_index](embedding).view((-1, self.num_options, self.num_actions[agent_index]))
        act_dist = Categorical(logits=F.log_softmax(x, dim=-1))

        if self.use_central_critic:
            if self.use_teamgrid:
                agent_values = x.view((-1, self.num_options, self.num_actions)) if self.use_act_values else x.view((-1, self.num_options))
            else:
                agent_values = x.view((-1, self.num_options, self.num_actions[agent_index])) if self.use_act_values else x.view(
                    (-1, self.num_options))

            # agent_target_values = x.view(
            #     (-1, self.num_options, self.num_actions)) if self.use_act_values else x.view((-1, self.num_options))

        else:
            if self.use_teamgrid:
                x = self.critic(embedding)
                agent_values = x.view((-1, self.num_options, self.num_actions)) if self.use_act_values else x.view((-1, self.num_options))
            else: #here critic is a list
                x = self.critic[agent_index](embedding)
                agent_values = x.view((-1, self.num_options, self.num_actions[agent_index])) if self.use_act_values else x.view(
                    (-1, self.num_options))

            # x = self.target_critic(embedding)
            # agent_target_values = x.view(
            #     (-1, self.num_options, self.num_actions)) if self.use_act_values else x.view((-1, self.num_options))


        #if self.num_options is not None:
        if self.use_term_fn:
            if self.use_teamgrid:
                x = self.term_fn(embedding).view((-1, self.num_options))
                term_dist = Bernoulli(probs=torch.sigmoid(x))
            else:
                x = self.term_fn[agent_index](embedding).view((-1, self.num_options))
                term_dist = Bernoulli(probs=torch.sigmoid(x))

        else:
            term_dist = None


        if not self.always_broadcast:
            if self.use_teamgrid:
                x_b = self.broadcast_net(embedding).view((-1, self.num_options, self.num_actions, self.num_broadcasts))
                broadcast_dist = Categorical(
                    logits=F.log_softmax(x_b, dim=-1))  # we need softmax on values depending on broadcast penalty

                agent_values_b = x_b.view((-1, self.num_options, self.num_actions, self.num_broadcasts)) if self.use_act_values else x_b.view((-1, self.num_options))
            else:
                x_b = self.broadcast_net[agent_index](embedding).view((-1, self.num_options, self.num_actions[agent_index], self.num_broadcasts))
                broadcast_dist = Categorical(
                    logits=F.log_softmax(x_b, dim=-1))  # we need softmax on values depending on broadcast penalty

                agent_values_b = x_b.view(
                    (-1, self.num_options, self.num_actions[agent_index], self.num_broadcasts)) if self.use_act_values else x_b.view((-1, self.num_options))



            if self.num_options is not None:
            # return act_dist, agent_values, agent_target_values, agent_values_b, agent_target_values_b, new_agent_memory, term_dist, broadcast_dist, embedding
            # if self.use_broadcasting and not self.always_broadcast:
                return act_dist, agent_values, agent_values_b, new_agent_memory, term_dist, broadcast_dist, embedding
            else:
                return act_dist, agent_values, agent_values_b, new_agent_memory, broadcast_dist, embedding

                #return act_dist, agent_values, new_agent_memory, term_dist, embedding
        else:
            if self.num_options is not None:
                return act_dist, agent_values, new_agent_memory, term_dist, embedding
            else:
                return act_dist, agent_values, new_agent_memory, embedding




    # def forward_central_critic(self, masked_embeddings, option_idxs, action_idxs, broadcast_idxs, coordinator_memory):
    def forward_central_critic(self, masked_embeddings, option_idxs, action_idxs, broadcast_idxs,
                                   coordinator_memory):
        if self.num_options is not None:
            option_onehots = []
            for option_idxs_j in option_idxs:
                option_onehots.append(utils.idx_to_onehot(option_idxs_j.long(), self.num_options))
        else:
            option_onehots = None

        if not self.use_teamgrid:
            #print('action_idx', action_idxs)
            action_onehots = []
            for i, action_idxs_j in enumerate(action_idxs):
                action_onehots.append(utils.idx_to_onehot(action_idxs_j.long(), self.num_actions[i])) # max(self.num_actions)
        else:
            action_onehots = []
            for action_idxs_j in action_idxs:
                action_onehots.append(utils.idx_to_onehot(action_idxs_j.long(), self.num_actions))


        if self.num_broadcasts == 1:
            broadcast_onehots = copy.deepcopy(broadcast_idxs)
            broadcast_onehots = [item.unsqueeze(1) for item in broadcast_onehots]
            # for i in range(len(broadcast_onehots)):
            #     broadcast_onehots[i] = broadcast_onehots[i].long()

            #print('broadcast_onehots',broadcast_onehots)
        else:
            broadcast_onehots = []
            #print('self.num_broadcasts', self.num_broadcasts)
            for broadcast_idxs_j in broadcast_idxs:
                broadcast_onehots.append(utils.idx_to_onehot(broadcast_idxs_j.long(), self.num_broadcasts))

       # print('star', *broadcast_onehots)
        if self.num_options is not None:
            # coordinator_embedding = torch.cat([*masked_embeddings, *option_onehots, *action_onehots, *broadcast_onehots], dim=1)
            coordinator_embedding = torch.cat(
                [*masked_embeddings, *option_onehots, *action_onehots, *broadcast_onehots], dim=1)
        else:
            # coordinator_embedding = torch.cat(
            #     [*masked_embeddings, *action_onehots, *broadcast_onehots], dim=1)
            coordinator_embedding = torch.cat(
                [*masked_embeddings, *action_onehots, *broadcast_onehots], dim=1)
        #print('coordinator_embedding', coordinator_embedding.size())

        if self.use_memory_coord:
            # hidden = (coordinator_memory[:, :self.semi_memory_size], coordinator_memory[:, self.semi_memory_size:])
            hidden = (coordinator_memory[:, :self.coord_semi_memory_size], coordinator_memory[:, self.coord_semi_memory_size:])
            hidden = self.coordinator_rnn(coordinator_embedding, hidden)
            coordinator_embedding = hidden[0]
            coordinator_memory = torch.cat(hidden, dim=1)

        assert self.use_central_critic
        value = self.critic(coordinator_embedding)
        #target_value = self.target_critic(coordinator_embedding)
        # value_a = torch.tensor(np.array(values)[:,0])
        # value_b = torch.tensor(np.array(values)[:,1])

        return coordinator_embedding, value.squeeze(), coordinator_memory.squeeze()

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def _embed_observation(self, obs, agent_memory, agent_index):
        if self.use_teamgrid:
            x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)

        else:
            x = obs.image


        if self.use_memory_agents:
            if self.use_teamgrid:
                hidden = (agent_memory[:, :self.semi_memory_size], agent_memory[:, self.semi_memory_size:])
                hidden = self.agent_memory_rnn(x, hidden)
                embedding = hidden[0]
                agent_memory = torch.cat(hidden, dim=1)
            else:
                hidden = (agent_memory[:, :self.semi_memory_size[agent_index]], agent_memory[:, self.semi_memory_size[agent_index]:])
                hidden = self.agent_memory_rnn[agent_index](x, hidden)
                embedding = hidden[0]
                agent_memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

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
