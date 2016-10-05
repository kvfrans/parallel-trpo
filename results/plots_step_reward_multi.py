import numpy as np
import matplotlib.pyplot as plt
import json
import sys

task = "Reacher-v1"

steps = []
rewards = []

for i in xrange(8):
    with open("speedup/"+task+"-"+str(i+1)) as data_file:
        data = json.load(data_file)

    steps.append([])
    rewards.append([])
    for j in xrange(len(data["mean_reward"])):
        rewards[i].append(data["mean_reward"][j])
        steps[i].append(j)

    plt.plot(np.array(steps[i]),np.array(rewards[i]),label="threads: "+str(i))

plt.xlabel("Steps")
plt.ylabel("Reward")
plt.legend(loc=4)
plt.title(task)
plt.show()
