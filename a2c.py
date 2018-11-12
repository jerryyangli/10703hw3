import sys
import argparse
import numpy as np
#import tensorflow as tf
#import keras
import gym
import torch.nn as nn
import torch
import time
import torch.nn.functional as F
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import datetime
import math

#from reinforce import Reinforce


class A2C(object):
	# Implementation of N-step Advantage Actor Critic.
	# This class inherits the Reinforce class, so for example, you can reuse
	# generate_episode() here.

	def __init__(self, model, lr, critic_model, critic_lr, n=20):
		# Initializes A2C.
		# Args:
		# - model: The actor model.
		# - lr: Learning rate for the actor model.
		# - critic_model: The critic model.
		# - critic_lr: Learning rate for the critic model.
		# - n: The value of N in N-step A2C.
		self.model = model
		self.critic_model = critic_model
		self.n = n
		# TODO: Define any training operations and optimizers here, initialize
		#       your variables, or alternately compile your model here.
		self.optimizer = torch.optim.Adam (self.model.parameters(), lr = lr)
		self.critic_optimizer = torch.optim.Adam (self.critic_model.parameters(), lr = critic_lr)  

	def train(self, env, num_train, gamma=1.0):
		# Trains the model on a single episode using A2C.
		# TODO: Implement this method. It may be helpful to call the class
		#       method generate_episode() to generate training data.
		for i in range(num_train):
			self.model.train()
			self.critic_model.train()
			states, actions, rewards, logprobs, score= self.generate_episode(env)
			rewards = np.array(rewards)
			T = len(states)
			x = torch.from_numpy(np.asarray(states).reshape(T, -1)).float()
			q_values = self.critic_model.forward(x)
			Vend_weighted = torch.zeros([T, 1])
			if self.n<T:
				Vend_weighted[0:T-self.n] = (gamma ** self.n) * q_values[self.n:T]
			rewards_weighted = torch.zeros([T, 1])
			weight = gamma ** np.array(range(self.n))
			index = np.minimum(T-np.array(range(T)), self.n)
			for i in range(T):
				rewards_weighted[i, 0] = (rewards[i:i+index[i]] * weight[0:index[i]]).sum()/100
			Rt = Vend_weighted + rewards_weighted
			with torch.no_grad():
				value_diff = torch.squeeze(Rt - q_values)
			loss_model =  -(torch.cat(logprobs) * value_diff).mean()
			self.optimizer.zero_grad()
			loss_model.backward()
			self.optimizer.step()
			loss_critic_model = ((Rt - q_values) ** 2).mean()
			self.critic_optimizer.zero_grad()
			loss_critic_model.backward()
			self.critic_optimizer.step()		

		return

	def test(self, env, num_test, gamma=1.0, render=False):
		reward_cumulatives=np.zeros(num_test)
		for episode in range(num_test):
			state_current=env.reset()
			if(render):
				env.render()
			self.model.eval()
			with torch.no_grad():
				is_terminal=0
				while(not is_terminal):
					state=torch.from_numpy(state_current.reshape(1,-1)).float()
					logits=self.model.forward(state)
					m = Categorical(logits=logits)
					action = m.sample().item()
					state_next, reward, is_terminal, debug_info = env.step(action)
					if(render):
						env.render()
					state_current=state_next
					reward_cumulatives[episode]+=reward
		reward_mean = np.mean(reward_cumulatives)
		reward_std = np.std(reward_cumulatives)
		return reward_mean,reward_std


	def generate_episode(self, env):
		# Generates an episode by executing the current policy in the given env.
		# Returns:
		# - a list of states, indexed by time step
		# - a list of actions, indexed by time step
		# - a list of rewards, indexed by time step
		states = []
		actions = []
		rewards = []
		logprobs=[]
		state_current=env.reset()
		is_terminal = 0
		score=0
		while(not is_terminal):
			state=torch.from_numpy(state_current.reshape(1,-1)).float()
			logits=self.model.forward(state)
			m = Categorical(logits=logits)
			action = m.sample()
			state_next, reward, is_terminal, debug_info = env.step(action.item())
			states.append(state_current)
			actions.append(action)
			rewards.append(reward)
			logprobs.append(m.log_prob(action))

			state_current=state_next
			score+=reward
		return states, actions, rewards, logprobs, score

def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	#parser.add_argument('--model-config-path', dest='model_config_path',
	#                    type=str, default='LunarLander-v2-config.json',
	#                    help="Path to the actor model config file.")
	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=50000, help="Number of episodes to train on.")
	parser.add_argument('--lr', dest='lr', type=float,
						default=5e-4, help="The actor's learning rate.")
	parser.add_argument('--critic-lr', dest='critic_lr', type=float,
						default=1e-4, help="The critic's learning rate.")
	parser.add_argument('--n', dest='n', type=int,
						default=20, help="The value of N in N-step A2C.")

	# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
							  action='store_true',
							  help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
							  action='store_false',
							  help="Whether to render the environment.")
	parser.set_defaults(render=False)

	return parser.parse_args()

class actor_network(nn.Module):
	def __init__(self):
		super(actor_network, self).__init__()
		self.number_of_features = 8
		self.number_of_actions = 4
		self.linear1 = nn.Linear(self.number_of_features, 16)
		self.linear2 = nn.Linear(16, 16)
		self.linear3 = nn.Linear(16, 16)
		self.linear4 = nn.Linear(16, self.number_of_actions)
		self.r = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax (dim = 1)

	def forward(self, x):
		x = self.r(self.linear1(x))
		x = self.r(self.linear2(x))
		x = self.r(self.linear3(x))
		x = self.linear4(x)
		x = self.logsoftmax(x)
		return x


class critic_network(nn.Module):
	def __init__(self):
		super(critic_network, self).__init__()
		self.number_of_features = 8
		self.linear1 = nn.Linear(self.number_of_features, 32)
		self.linear2 = nn.Linear(32, 32)
		self.linear3 = nn.Linear(32, 32)
		self.linear4 = nn.Linear(32, 1)
		self.r = nn.ReLU()

	def forward(self, x):
		x = self.r(self.linear1(x))
		x = self.r(self.linear2(x))
		x = self.r(self.linear3(x))
		x = self.linear4(x)
		return x

def weights_init(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_ (m.weight.data)


def main(args):
	# Parse command-line arguments.
	args = parse_arguments()
	#model_config_path = args.model_config_path
	num_episodes = args.num_episodes
	lr = args.lr
	critic_lr = args.critic_lr
	n = args.n
	render = args.render
	num_train_per_round = 500
	num_test_per_round = 100
	num_round = math.ceil(num_episodes / num_train_per_round)

	# Create the environment.
	env = gym.make('LunarLander-v2')
	seed = 777
	np.random.seed(seed)
	env.seed(seed)
	torch.manual_seed(seed)
	model = actor_network()
	critic_model = critic_network()
	model.apply(weights_init)
	critic_model.apply(weights_init)
	agent = A2C(model, lr, critic_model, critic_lr, n=n)

	# TODO: Train the model using A2C and plot the learning curves.
	result = np.zeros([num_round, 2])
	for i in range(num_round):
		agent.train(env, num_train_per_round, gamma = 1)
		result[i, 0], result[i, 1] = agent.test(env, num_test_per_round, gamma = 1)
		print ("{0}\t{1}" .format(result[i, 0], result[i, 1]))    


if __name__ == '__main__':
	main(sys.argv)
