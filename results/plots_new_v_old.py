import numpy as np
import matplotlib.pyplot as plt
import json
import sys

task = "Reacher-v1"

times = []
rewards = []
t = []
r = []

trials = ["HalfCheetah-newmethod","HalfCheetah-oldmethod"]
for i in xrange(2):
    with open("new_vs_old/"+trials[i]) as data_file:
        data = json.load(data_file)

    times.append([])
    rewards.append([])
    totaltime = 0
    for e in xrange(len(data["mean_reward"])):
        totaltime += data["timesteps"][e]
        # totaltime += 1
        if i == 0:
            times[i].append(totaltime)
        else:
            times[i].append(totaltime)
        rewards[i].append(data["mean_reward"][e])

    t.append(np.array(times[i]))
    r.append(np.array(rewards[i]))

    if i == 0:
        plt.plot(t[i],r[i],color=(1 - (i/2.0),i/2.0,1.0),label="New Method")
    else:
        plt.plot(t[i],r[i],color=(1 - (i/2.0),i/2.0,1.0),label="Old Method")
plt.xlabel("Environment Steps Seen")
plt.ylabel("Average return")
plt.legend(loc=4)
plt.title(task)
plt.show()
