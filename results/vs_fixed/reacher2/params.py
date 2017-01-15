import numpy as np
import matplotlib.pyplot as plt
import json
import sys

times = []
rewards = []
t = []
r = []

trials = ["Reacher-v1-adaptive-margin-1000.000000-0.001000-300.000000-0.000500"]
names = ["Adapt both w/ margin"]
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
        # import ipdb; ipdb.set_trace()
        avg += data["timesteps"][e]
        avgcount += 1

        if time_since > 10000 and totaltime < 4000000:
            time_since = 0
            # totaltime += 1
            if i == 0:
                times[i].append(totaltime)
            else:
                times[i].append(totaltime)
            rewards[i].append(avg/avgcount)

            avg = 0
            avgcount = 0

    t.append(np.array(times[i]))
    r.append(np.array(rewards[i]))

    lin, = plt.plot(t[i],r[i],label=names[i])
    # if i == 0:
        # lin.remove()


plt.xlabel("Environment Steps Seen")
plt.ylabel("Average return")
leg = plt.legend(loc=4)
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
plt.title("Reacher-v1")
plt.show()
