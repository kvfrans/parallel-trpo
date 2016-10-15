import numpy as np
import matplotlib.pyplot as plt
import json
import sys

task = "Reacher-v1"

times = []
rewards = []
totaltime = []
t = []
r = []

trials = ["10xKL","20000"]
for i in xrange(2):
    with open("montecarlo-test/"+trials[i]) as data_file:
        data = json.load(data_file)

    times.append([])
    rewards.append([])
    for e in xrange(len(data["mean_reward"])):
        if i == 0:
            times[i].append(e)
        else:
            times[i].append(e)
        rewards[i].append(data["mean_reward"][e])

    t.append(np.array(times[i]))
    r.append(np.array(rewards[i]))

    if i == 0:
        plt.plot(t[i],r[i],color=(1 - (i/2.0),i/2.0,1.0),label="0.01 max KL")
    else:
        plt.plot(t[i],r[i],color=(1 - (i/2.0),i/2.0,1.0),label="0.001 max KL")
plt.xlabel("Environment Steps Seen")
plt.ylabel("Average return")
plt.legend(loc=4)
plt.title(task)
plt.show()
