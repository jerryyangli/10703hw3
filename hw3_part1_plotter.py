import numpy as np
import matplotlib.pyplot as plt

file = open("out_P1Q2.txt", "r")
count = 0
mean=[]
stds=[]
for line in file:
	if (count-1) %3 ==0:
		# print(line)
		_, m, s = line.split()
		# print(reward1,reward2)
		mean.append(float(m))
		stds.append(float(s))
	count+=1

episodes_per_time_point=200
episodes=np.arange(0,(len(mean))*episodes_per_time_point,episodes_per_time_point)

plt.errorbar(episodes, mean, stds, marker='*',color='green',ecolor='blue')
plt.axhline(y=200, color='r', linestyle=':')

plt.title("Reinforce: Cumulative Test Reward vs Training Episodes")
plt.xlabel('Training Episode')
plt.ylabel('Cumulative Test Reward')
plt.show()
