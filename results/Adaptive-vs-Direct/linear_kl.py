import numpy as np
import matplotlib.pyplot as plt
import json
import sys

task = "HalfCheetah-v1"

times = []
rewards = []
t = []
r = []

trials = ["Reacher-v1-linear-0.000001-0.000000","Reacher-v1-linear-0.000010-0.000000","Reacher-v1-linear-0.000100-0.000000"]
for i in xrange(len(trials)):
    with open(trials[i]) as data_file:
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

    plt.plot(t[i],r[i],color=(1 - (i/4.0),i/4.0,1.0),label=trials[i])

plt.xlabel("Environment Steps Seen")
plt.ylabel("Average return")
plt.legend(loc=4)
plt.title(task)
plt.show()
