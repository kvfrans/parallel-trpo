import numpy as np
import tensorflow as tf
import gym
from utils import *
from model import *
import argparse
from rollouts import *
import json



parser = argparse.ArgumentParser(description='Test the new good lib.')
parser.add_argument("--task", type=str, default='Reacher-v1')
parser.add_argument("--eps_per_batch", type=int, default=400)
parser.add_argument("--max_pathlength", type=int, default=50)
parser.add_argument("--n_iter", type=int, default=350)
parser.add_argument("--gamma", type=float, default=.99)
parser.add_argument("--max_kl", type=float, default=.001)
parser.add_argument("--cg_damping", type=float, default=1e-3)
parser.add_argument("--num_threads", type=int, default=3)
parser.add_argument("--monitor", type=bool, default=False)
args = parser.parse_args()


learner_tasks = multiprocessing.JoinableQueue()
learner_results = multiprocessing.Queue()
learner_env = gym.make(args.task)

learner = TRPO(args, learner_env.observation_space, learner_env.action_space, learner_tasks, learner_results)
learner.start()
rollouts = ParallelRollout(args)

learner_tasks.put(1)
learner_tasks.join()
starting_weights = learner_results.get()
rollouts.set_policy_weights(starting_weights)

start_time = time.time()
history = {}
history["rollout_time"] = []
history["learn_time"] = []
history["mean_reward"] = []

for iteration in xrange(args.n_iter):

    # runs a bunch of async processes that collect rollouts
    rollout_start = time.time()
    paths = rollouts.rollout()
    rollout_time = (time.time() - rollout_start) / 60.0

    # Why is the learner in an async process?
    # Well, it turns out tensorflow has an issue: when there's a tf.Session in the main thread
    # and an async process creates another tf.Session, it will freeze up.
    # To solve this, we just make the learner's tf.Session in its own async process,
    # and wait until the learner's done before continuing the main thread.
    learn_start = time.time()
    learner_tasks.put(paths)
    learner_tasks.join()
    new_policy_weights, mean_reward = learner_results.get()
    learn_time = (time.time() - learn_start) / 60.0
    print "-------- Iteration %d ----------" % iteration
    print "Total time: %.2f mins" % ((time.time() - start_time) / 60.0)

    history["rollout_time"].append(rollout_time)
    history["learn_time"].append(learn_time)
    history["mean_reward"].append(mean_reward)

    if iteration % 100 == 0:
        with open(args.task + "-" + str(args.num_threads), "w") as outfile:
            json.dump(history,outfile)

    rollouts.set_policy_weights(new_policy_weights)

rollouts.end()
