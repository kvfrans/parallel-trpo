import numpy as np
import matplotlib.pyplot as plt
import json
import sys

# task = "Reacher-v1"
task = sys.argv[1]

with open("done/"+task+"-1") as data_file:
    data_1 = json.load(data_file)

with open("done/"+task+"-5") as data_file:
    data_2 = json.load(data_file)

times_1 = []
rewards_1 = []
totaltime_1 = 0
for i in xrange(len(data_1["mean_reward"])):
    totaltime_1 += data_1["rollout_time"][i] + data_1["learn_time"][i]
    times_1.append(totaltime_1)
    rewards_1.append(data_1["mean_reward"][i])

times_2 = []
rewards_2 = []
totaltime_2 = 0
for i in xrange(len(data_2["mean_reward"])):
    totaltime_2 += data_2["rollout_time"][i] + data_2["learn_time"][i]
    times_2.append(totaltime_2)
    rewards_2.append(data_2["mean_reward"][i])

t1 = np.array(times_1)
r1 = np.array(rewards_1)
t2 = np.array(times_2)
r2 = np.array(rewards_2)

# t1 = np.arange(len(times_1))
# t2 = np.arange(len(times_2))

plt.plot(t1,r1,"r",label="single thread")
plt.plot(t2,r2,"b",label="5 threads")
plt.xlabel("Training time (minutes)")
plt.ylabel("Average return")
plt.legend(loc=4)
plt.title(task)
plt.show()
