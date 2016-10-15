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

trials = [1000,5000,10000,20000]
for i in xrange(4):
    with open("montecarlo-test/"+str(trials[i])) as data_file:
        data = json.load(data_file)

    times.append([])
    rewards.append([])
    for e in xrange(len(data["mean_reward"])):
        times[i].append(e*trials[i])
        rewards[i].append(data["mean_reward"][e])

    t.append(np.array(times[i]))
    r.append(np.array(rewards[i]))

    plt.plot(t[i],r[i],color=(1 - (i/8.0),i/8.0,1.0),label=("%d steps / update" % (trials[i])))
plt.xlabel("Environment Steps Seen")
plt.ylabel("Average return")
plt.legend(loc=4)
plt.title(task)
plt.show()
