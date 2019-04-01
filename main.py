import numpy as np
from network.broadcast_net import *

'''import action_policy_net (or, pi_net) and broadcast_net that are trained elsewhere. E.g., 

pi_nets = {agent:pi_net for agent in range(n_agent)}
where

pi_net can be of the form 
for agent in range(n_agent):
	pi_nets[agent] = lambda: VanillaNet_LR(agent_action_dim, FCBody(agent_state_dim))
'''


episodes = 100

n_agent = 3


agent_states = [j for j in range(13)] # modify this
agent_state_dim = len(agent_states)

joint_state_list = [s for s in list(itertools.product(agent_states, repeat=n_agent))
                            if len(s) == len(np.unique(s))]

agent_action_dim = 4
agent_actions = [j for j in range(agent_action_dim)]

#optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)

# inital beliefs
belief_pmfs = {agent:np.random.uniform(agent_state_dim) for agent in range(n_agent)} # subject to change


# initial likelihood
likelihood = {agent:np.random.uniform(agent_state_dim) for agent in range(n_agent)} # subject to change


# broadcst = Broadcast_net() imported from broadcast_net.py
broadcast_nets = {agent:broadcast for agent in range(n_agent)} #initialize broadcast nets for all agents

def train_agent(episodes):
    #running_reward = 10
    for episode in range(episodes):
        
        
        joint_state = env.reset() # Reset environment and record the starting state
        done = False 
        
        for iteration in range(1000):
            
            for agent in range(n_agent):
            	
            	agent_state = joint_state[agent]

            	agent_action = select_action(agent_state)
	            agent_broadcast_action = select_broadcast(agent_state)

            	# update factored belief
            	likelihood[agent] = Likelihood_fn(pi_nets[agent], broadcast_nets[agent], agents_states, agent_actions).update(belief_pmfs[agent])
		
				belief_pmfs[agent] = [j*k for (j,k) in zip(belief_pmfs[agent],likelihood[agent])]

				belief_pmfs[agent] /= np.sum(belief_pmfs[agent]) # normalized

				remarginalized_belief[agent] = remarginalize_belief(agent, belief_pmfs, joint_obs, likelihood[agent], agent_states)
            
	            #sample agent_state according to agent's belief pmf
	            sample_agent_state = np.random.choice(env.cell_list,p=remarginalized_belief[agent])
	            
	            # TODO: import pi_nets
	            # sample agent_action from pi_nets[agent]
	           
	            # Step through environment using chosen action
	            agent_next_state, reward, done, _ = env.step(agent_action.data[0])

	            error_due_to_no_broadcast = np.linalg.norm([i-j for (i,j) in zip(self.env.tocellcoord[sample_agent_state],self.env.tocellcoord[agent_state])])

				selfishness_penalty = env.selfishness_penalty*error_due_to_no_broadcast #*(error_due_to_no_broadcast > self.no_broadcast_threshold)

	            # subject to change
	            reward += env.broadcast_penalty*agent_broadcast_action + selfishness_penalty*(1-agent_broadcast_action)
	            
	            # Save reward
	            broadcast.reward_episode.append(reward)
	            if done:
	                break

                agent_state = agent_next_state
                next_joint_state[agent] = agent_state
            joint_state = tuple(next_joint_state[agent])      
	#         # Used to determine when the environment is solved.
	#         running_reward = (running_reward * 0.99) + (time * 0.01)
	            
        update_broadcast_policy(broadcast_nets[agent])
        curr_comm_bel_vec = [common_belief(joint_state,belief_pmfs) for joint_state in joint_state_list] # We may not need this pmf

		
	

#_________________________________________________________________________________________________________

def common_belief(joint_state, belief_pmfs):
	res = 1.0
	for j in range(len(joint_state)):
		res *= belief_pmfs[j][joint_state[j]]
	return res


class Likelihood_fn:
	def __init__(self, pi_net, broadcast_net, agents_states, agent_actions):
		self.pi_net = pi_net
		self.broadcast_net = broadcast_net
		self.agent_states = agent_states
		self.agent_actions = agent_actions

	def update(belief_pmf):
		return [l*(k/m) for (l,k,m) in zip(Likelihood_fn(pi_net, broadcast_net, agents_states, agent_actions), sample_average(belief_pmf, pi_net, broadcast_net, agent_states, agent_actions), (sample_average(belief_pmf, pi_net, broadcast_net, agent_states)))]

def sample_average(belief_pmf, pi_net, broadcast_net, agent_states, agent_actions = None, num_samples = 1000):
	counts_tuple = {i:0 for i in list(itertools.product(agent_states,agent_actions,[0,1]))}
	counts_state = {i:0 for i in agent_states}
	for _ in range(num_samples):
		sample_agent_state = np.random.choice(agent_states, p = belief_pmf)
		counts_state[sample_agent_state] += 1

		if agent_actions is not None:
			sample_agent_action = np.random.choice(agent_actions, p = pi_net[agent].forward(agent_actions))
			sample_agent_broadcast = np.random.choice([0,1], p = broadcast_net.forward(agent_states))

			sample_tuple = tuple(sample_agent_state,sample_agent_action,sample_agent_broadcast)
			counts_tuple[sample_tuple] += 1

			return list(counts_tuple.values())/num_samples
		else:
			return list(counts_state.values())/num_samples	



def remarginalize_belief(agent, belief, obs, likelihood, agents_states, num_samples = 1000, iterations = 100):
    for i in range(iterations):
        vector = belief[agent]
        for _ in range(num_samples):
            sample_state = np.random.choice(agent_states, p = belief[agent])

            other_agents = [ag for ag in range(len(obs)) if ag != agent]
            prob = 1.0
            for ag in other_agents:
                if obs[ag] != None: 
                    prob *= 1- 1/agent_states # prob that obs[ag] != obs[agent]
                else:
                    samples_oth_agent = np.choice(agent_states, num_sample, p = belief[ag]) 
                    prob *= np.sum([s != sample_state for s in samples_oth_agent])/num_sample

            vector[sample_state] = likelihood*prob
        belief[agent] /= vector/np.sum(vector) #normalized
    return belief[agent]
