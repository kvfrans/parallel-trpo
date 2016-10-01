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
for i in xrange(8):
    with open("speedup/"+task+"-"+str(i+1)) as data_file:
        data = json.load(data_file)

    times.append([])
    rewards.append([])
    totaltime.append(0)
    for e in xrange(len(data["mean_reward"])):
        totaltime[i] += data["rollout_time"][e] + data["learn_time"][e]
        times[i].append(totaltime[i])
        rewards[i].append(data["mean_reward"][e])

    t.append(np.array(times[i]))
    r.append(np.array(rewards[i]))

    plt.plot(t[i],r[i],color=(1 - (i/8.0),i/8.0,1.0),label=("%d threads" % (i+1)))
plt.xlabel("Training time (minutes)")
plt.ylabel("Average return")
plt.legend(loc=4)
plt.title(task)
plt.show()
