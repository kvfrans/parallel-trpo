import numpy as np
import matplotlib.pyplot as plt
import json
import sys

task = "Reacher-v1"

times = []
rewards = []
totaltime = []
t = []
l = []
r = []
for i in xrange(8):
    with open("speedup/"+task+"-"+str(i+1)) as data_file:
        data = json.load(data_file)


    t.append(np.mean(data["rollout_time"]))
    l.append(np.mean(data["learn_time"]))
    r.append(i+1)

print r
print t
plt.plot(r,t,label="Rollout time")
plt.plot(r,l,label="Learning time")
plt.xlabel("Threads")
plt.ylabel("Time for an iteration (seconds)")
plt.legend(loc=1)
plt.title(task)
plt.show()
