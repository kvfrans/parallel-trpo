import numpy as np
import matplotlib.pyplot as plt
import json
import sys

times = []
rewards = []
t = []
r = []

trials = ["HalfCheetah-v1-none-20000.000000-0.001000-0.000000-0.000000","HalfCheetah-v1-none-20000.000000-0.005000-0.000000-0.000000","HalfCheetah-v1-none-20000.000000-0.010000-0.000000-0.000000"]
names = ["Fixed 0.001 KL", "Fixed 0.005 KL", "Fixed 0.01 KL"]

for i in xrange(len(trials)):
    with open(trials[i]) as data_file:
        data = json.load(data_file)

    times.append([])
    rewards.append([])
    totaltime = 0

    time_since = 0
    avg = 0
    avgcount = 0

    for e in xrange(len(data["mean_reward"])):
        totaltime += data["timesteps"][e]

        time_since += data["timesteps"][e]
        avg += data["mean_reward"][e]
        avgcount += 1

        if time_since > 10000:
            time_since = 0
            # totaltime += 1
            if i == 0:
                times[i].append(totaltime)
            else:
                times[i].append(totaltime)
            rewards[i].append(data["mean_reward"][e])

            avg = 0
            avgcount = 0

    t.append(np.array(times[i]))
    r.append(np.array(rewards[i]))

    plt.plot(t[i],r[i],color=(1 - (i/5.0),i/5.0,1.0),label=names[i])

plt.xlabel("Environment Steps Seen")
plt.ylabel("Average return")
plt.legend(loc=4)
plt.title("HalfCheetah-v1")
plt.show()
