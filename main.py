import numpy as np
import tensorflow as tf
import gym
from utils import *
from model import *
import argparse
from rollouts import *
import json



parser = argparse.ArgumentParser(description='TRPO.')
# these parameters should stay the same
parser.add_argument("--task", type=str, default='Reacher-v1')
parser.add_argument("--timesteps_per_batch", type=int, default=10000)
parser.add_argument("--n_steps", type=int, default=6000000)
parser.add_argument("--gamma", type=float, default=.99)
parser.add_argument("--max_kl", type=float, default=.001)
parser.add_argument("--cg_damping", type=float, default=1e-3)
parser.add_argument("--num_threads", type=int, default=5)
parser.add_argument("--monitor", type=bool, default=False)

# change these parameters for testing
parser.add_argument("--decay_method", type=str, default="adaptive") # adaptive, none
parser.add_argument("--timestep_adapt", type=int, default=0)
parser.add_argument("--kl_adapt", type=float, default=0)

args = parser.parse_args()
args.max_pathlength = gym.spec(args.task).timestep_limit

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
history["timesteps"] = []

# start it off with a big negative number
last_reward = -1000000
recent_total_reward = 0

totalsteps = 0;

starting_timesteps = args.timesteps_per_batch
starting_kl = args.max_kl

iteration = 0
while True:
    iteration += 1;

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
    learner_tasks.put((2,args.max_kl))
    learner_tasks.put(paths)
    learner_tasks.join()
    new_policy_weights, mean_reward = learner_results.get()
    learn_time = (time.time() - learn_start) / 60.0
    print "-------- Iteration %d ----------" % iteration
    print "Total time: %.2f mins" % ((time.time() - start_time) / 60.0)

    history["rollout_time"].append(rollout_time)
    history["learn_time"].append(learn_time)
    history["mean_reward"].append(mean_reward)
    history["timesteps"].append(args.timesteps_per_batch)
    history["maxkl"].append(args.max_kl)

    recent_total_reward += mean_reward

    if args.decay_method == "adaptive":
        if iteration % 10 == 0:
            if recent_total_reward < last_reward:
                print "Policy is not improving. Decrease KL and increase steps."
                if args.timesteps_per_batch < 20000:
                    args.timesteps_per_batch += args.timestep_adapt
                if args.max_kl > 0.001:
                    args.max_kl -= args.kl_adapt
            else:
                print "Policy is improving. Increase KL and decrease steps."
                if args.timesteps_per_batch > 1200:
                    args.timesteps_per_batch -= args.timestep_adapt
                if args.max_kl < 0.01:
                    args.max_kl += args.kl_adapt
            last_reward = recent_total_reward
            recent_total_reward = 0


    if args.decay_method == "adaptive-margin":
        if iteration % 10 == 0:
            scaled_last = last_reward + abs(last_reward * 0.05)
            print "Last reward: %f Scaled: %f Recent: %f" % (last_reward, scaled_last, recent_total_reward)
            if recent_total_reward < scaled_last:
                print "Policy is not improving. Decrease KL and increase steps."
                if args.timesteps_per_batch < 10000:
                    args.timesteps_per_batch += args.timestep_adapt
                if args.max_kl > 0.001:
                    args.max_kl -= args.kl_adapt
            else:
                print "Policy is improving. Increase KL and decrease steps."
                if args.timesteps_per_batch > 1200:
                    args.timesteps_per_batch -= args.timestep_adapt
                if args.max_kl < 0.01:
                    args.max_kl += args.kl_adapt
            last_reward = recent_total_reward
            recent_total_reward = 0

    print "Current steps is " + str(args.timesteps_per_batch) + " and KL is " + str(args.max_kl)

    if iteration % 100 == 0:
        with open("%s-%s-%f-%f-%f-%f" % (args.task, args.decay_method, starting_timesteps, starting_kl, args.timestep_adapt, args.kl_adapt), "w") as outfile:
            json.dump(history,outfile)

    totalsteps += args.timesteps_per_batch
    print "%d total steps have happened" % totalsteps
    if totalsteps > args.n_steps:
        break

    rollouts.set_policy_weights(new_policy_weights)

rollouts.end()
