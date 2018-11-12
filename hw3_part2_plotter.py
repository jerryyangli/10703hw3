import numpy as np
import matplotlib.pyplot as plt


file = open("a2c_n1", "r")
mean=[]
stds=[]
for line in file:
		m, s = line.split('\t')
		mean.append(float(m))
		stds.append(float(s))

episodes_per_time_point=500
episodes=np.arange(0,(len(mean))*episodes_per_time_point,episodes_per_time_point)

plt.errorbar(episodes, mean, stds, marker='*',color='green',ecolor='blue')
plt.axhline(y=200, color='r', linestyle=':')

plt.title("a2c (n=1): Cumulative Test Reward vs Training Episodes")
plt.xlabel('Training Episode')
plt.ylabel('Cumulative Test Reward')
plt.show()

file = open("a2c_n20", "r")
mean=[]
stds=[]
for line in file:
		m, s = line.split('\t')
		mean.append(float(m))
		stds.append(float(s))

episodes_per_time_point=500
episodes=np.arange(0,(len(mean))*episodes_per_time_point,episodes_per_time_point)

plt.errorbar(episodes, mean, stds, marker='*',color='green',ecolor='blue')
plt.axhline(y=200, color='r', linestyle=':')

plt.title("a2c (n=20): Cumulative Test Reward vs Training Episodes")
plt.xlabel('Training Episode')
plt.ylabel('Cumulative Test Reward')
plt.show()

file = open("a2c_n50", "r")
mean=[]
stds=[]
for line in file:
		m, s = line.split('\t')
		mean.append(float(m))
		stds.append(float(s))

episodes_per_time_point=500
episodes=np.arange(0,(len(mean))*episodes_per_time_point,episodes_per_time_point)

plt.errorbar(episodes, mean, stds, marker='*',color='green',ecolor='blue')
plt.axhline(y=200, color='r', linestyle=':')

plt.title("a2c (n=50): Cumulative Test Reward vs Training Episodes")
plt.xlabel('Training Episode')
plt.ylabel('Cumulative Test Reward')
plt.show()

file = open("a2c_n100", "r")
mean=[]
stds=[]
for line in file:
		m, s = line.split('\t')
		mean.append(float(m))
		stds.append(float(s))

episodes_per_time_point=500
episodes=np.arange(0,(len(mean))*episodes_per_time_point,episodes_per_time_point)

plt.errorbar(episodes, mean, stds, marker='*',color='green',ecolor='blue')
plt.axhline(y=200, color='r', linestyle=':')

plt.title("a2c (n=100): Cumulative Test Reward vs Training Episodes")
plt.xlabel('Training Episode')
plt.ylabel('Cumulative Test Reward')
plt.show()