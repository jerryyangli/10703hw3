import sys
import argparse
import numpy as np
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

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr,load_path=None):
        # Initializes Reinforce.
        # Args:
        # - model: The Reinforce model
        # - lr: Learning rate for the amodel
        # - load_path: load weights

        self.model = model
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=lr)
        if load_path is not None:
            self.model.load_state_dict(
                torch.load(
                    load_path,
                    map_location=lambda storage,
                    loc: storage))

    def train(self, env, gamma=1.0,num_episodes=50000):
        # Trains the model on a given number of episodes using REINFORCE.
        # Test the model every 200 episodes
        # Args:
        # - gamma: discount factor
        # - num_episodes: number of training episodes

        loss_avg=0
        device = torch.device('cpu')
        number_of_trainig_episodes_per_test=200
        for episode in range(num_episodes):
            self.model.train()
            self.model.to(device)
            self.optimizer.zero_grad()
            states,actions,rewards2,logprobs,score=self.generate_episode(env,device)
            rewards=np.array(rewards2)
            T=len(states)
            G_t=np.zeros(T)
            G_t[T-1]=rewards[T-1]
            for t in reversed(range(T-1)):
                G_t[t]=rewards[t:t+1]+gamma*G_t[t+1]
            G_t=torch.tensor(G_t/100).float().to(device)
            loss=-(torch.cat(logprobs)*G_t).mean()#/T
            loss.backward()
            loss_avg+=loss.item()
            self.optimizer.step()
            if(episode%number_of_trainig_episodes_per_test==0): 
                print('Episode: ', episode)
                reward_mean,reward_std=self.test(env,device,render=True)
                print('Loss: ', loss_avg/number_of_trainig_episodes_per_test)
                loss_avg=0

            if(episode%1000==0 and episode > 10000):
                self.save_model(datetime.datetime.now().isoformat())


    def test(self,env,device,render=False):
        # Run 100 test episodes and return mean and standard deviation of the reward

        number_of_episodes=100
        reward_cumulatives=np.zeros(number_of_episodes)
        avg_frames=np.zeros(number_of_episodes)
        for episode in range(number_of_episodes):
            state_current=env.reset()
            self.model.eval()
            with torch.no_grad():
                is_terminal=0
                count=0
                while(not is_terminal):
                    count+=1
                    state=torch.from_numpy(state_current.reshape(1,-1)).float().to(device)
                    logits=self.model.forward(state)
                    m = Categorical(logits=logits)
                    action = m.sample().item()
                    state_next, reward, is_terminal, debug_info = env.step(action)
                    state_current=state_next
                    reward_cumulatives[episode]+=reward
                avg_frames[episode]=count
        reward_mean = np.mean(reward_cumulatives)
        reward_std = np.std(reward_cumulatives)
        print('Testing: ', reward_mean ,reward_std)
        # print('average number of frames', np.mean(avg_frames))
        return reward_mean,reward_std


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def generate_episode(self, env, device, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # - a list of lopprobs, indexed by time step
        # - total reward of the episode

        states = []
        actions = []
        rewards = []
        logprobs=[]
        state_current=env.reset()
        is_terminal = 0
        score=0
        while(not is_terminal):
            #https://pytorch.org/docs/stable/_modules/torch/distributions/categorical.html
            state=torch.from_numpy(state_current.reshape(1,-1)).float().to(device)
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
        return states, actions, rewards,logprobs,score

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

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

class Reinforce_Network(nn.Module):
    def __init__(self):
        super(Reinforce_Network, self).__init__()
        self.number_of_features=8 
        self.number_of_actions=4

        self.linear1=nn.Linear(self.number_of_features, 16)
        self.linear2=nn.Linear(16,16)
        self.linear3=nn.Linear(16,16)
        self.linear4=nn.Linear(16,self.number_of_actions)

        self.r=nn.ReLU()
        self.logsoftmax=nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.r(self.linear1(x))
        x = self.r(self.linear2(x))
        x = self.r(self.linear3(x))
        x = self.linear4(x)
        x = self.logsoftmax(x)

        return x


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m,nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    #Select seed for comparison
    seed = 777
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)

    #create model and trainer and initialize them
    model=Reinforce_Network()
    model.apply(weights_init)
    trainer=Reinforce(model,lr)

    #Go!
    trainer.train(env,gamma=1)

if __name__ == '__main__':
    main(sys.argv)
