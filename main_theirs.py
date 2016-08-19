import numpy as np
import tensorflow as tf
import gym
from utils import *
from model import *
import argparse
from rollouts import *



parser = argparse.ArgumentParser(description='Test the new good lib.')
parser.add_argument("--task", type=str, default='Reacher-v1')
parser.add_argument("--timesteps_per_batch", type=int, default=1000)
parser.add_argument("--max_pathlength", type=int, default=49)
parser.add_argument("--n_iter", type=int, default=250)
parser.add_argument("--gamma", type=float, default=.99)
parser.add_argument("--max_kl", type=float, default=.001)
parser.add_argument("--cg_damping", type=float, default=1e-3)
parser.add_argument("--num_threads", type=int, default=3)
args = parser.parse_args()


learner_tasks = multiprocessing.JoinableQueue()
learner_results = multiprocessing.Queue()
learner_env = gym.make(args.task)

learner = TRPO(args, learner_env.observation_space, learner_env.action_space, learner_tasks, learner_results, learner_env)
learner.start()
rollouts = ParallelRollout(args)

learner_tasks.put(1)
learner_tasks.join()
# starting_weights = learner_results.get()
# rollouts.set_policy_weights(starting_weights)

start_time = time.time()
for iteration in xrange(args.n_iter):
    # runs a bunch of async processes that collect rollouts
    # paths = rollouts.rollout()

    # Why is the learner in an async process?
    # Well, it turns out tensorflow has an issue: when there's a tf.Session in the main thread
    # and an async process creates another tf.Session, it will freeze up.
    # To solve this, we just make the learner's tf.Session in its own async process,
    # and wait until the learner's done before continuing the main thread.
    # learner_tasks.put(paths)
    learner_tasks.put(0)
    learner_tasks.join()
    # new_policy_weights = learner_results.get()
    print "-------- Iteration %d ----------" % iteration
    print "Total time: %.2f mins" % ((time.time() - start_time) / 60.0)

    # rollouts.set_policy_weights(new_policy_weights)
