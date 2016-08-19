import numpy as np
import tensorflow as tf
import gym
from utils import *
from rollouts import *
from value_function import *
import time
import os
import logging
import random
import multiprocessing
import prettytensor as pt

class TRPO(multiprocessing.Process):
    def __init__(self, args, observation_space, action_space, task_q, result_q, env):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args
        self.env = env

    def makeModel(self):
        self.observation_size = self.observation_space.shape[0]
        self.action_size = np.prod(self.action_space.shape)
        self.hidden_size = 64

        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        self.session = tf.Session()

        self.obs = obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = advant = tf.placeholder(tf.float32, [None])
        self.oldaction_dist_mu = oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.oldaction_dist_logstd = oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])
        dtype = tf.float32

        with tf.variable_scope("policy"):
            h1 = fully_connected(self.obs, self.observation_size, self.hidden_size, weight_init, bias_init, "policy_h1")
            h1 = tf.nn.relu(h1)
            # h2 = fully_connected(h1, self.hidden_size, self.hidden_size, weight_init, bias_init, "policy_h2")
            # h2 = tf.nn.relu(h2)
            h3 = fully_connected(h1, self.hidden_size, self.action_size, weight_init, bias_init, "policy_h3")
            action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, self.action_size)).astype(np.float32), name="policy_logstd")
        # means for each action
        action_dist_mu = h3
        # log standard deviations for each actions
        action_dist_logstd = tf.tile(action_dist_logstd_param, tf.pack((tf.shape(action_dist_mu)[0], 1)))

        eps = 1e-8
        self.action_dist_mu = action_dist_mu
        self.action_dist_logstd = action_dist_logstd
        N = tf.shape(obs)[0]
        # compute probabilities of current actions and old action
        log_p_n = gauss_log_prob(action_dist_mu, action_dist_logstd, action)
        log_oldp_n = gauss_log_prob(oldaction_dist_mu, oldaction_dist_logstd, action)

        # proceed as before, good.
        ratio_n = tf.exp(log_p_n - log_oldp_n)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        var_list = tf.trainable_variables()

        # Introduced the change into here:
        kl = gauss_KL(oldaction_dist_mu, oldaction_dist_logstd,
                      action_dist_mu, action_dist_logstd) / Nf
        ent = gauss_ent(action_dist_mu, action_dist_logstd) / Nf

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = gauss_selfKL_firstfixed(action_dist_mu, action_dist_logstd) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.session.run(tf.initialize_variables(var_list))
        self.vf = LinearVF()

        self.get_policy = GetPolicyWeights(self.session, var_list)

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd  = \
            self.session.run([self.action_dist_mu, self.action_dist_logstd], {self.obs: obs})

        act = action_dist_mu + np.exp(action_dist_logstd)*np.random.randn(*action_dist_logstd.shape)

        return act.ravel(), \
            {"action_dist_mu": action_dist_mu,
                  "action_dist_logstd": action_dist_logstd}

    def run(self):
        self.makeModel()
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
            else:
                self.learn(paths)
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
        return

    def learn(self, paths):
        config = self.args
        self.advant = self.advantage

        paths = rollout_contin(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch,
                render = False) #(i % render_freq) == 0)

    # Computing returns and estimating advantage function.
        for path in paths:
            path["baseline"] = self.vf.predict(path)
            path["returns"] = discount(path["rewards"], config.gamma)
            path["advant"] = path["returns"] - path["baseline"]

        # Updating policy.
        action_dist_mu = np.concatenate([path["action_dists_mu"] for path in paths])
        action_dist_logstd = np.concatenate([path["action_dists_logstd"] for path in paths])
        obs_n = np.concatenate([path["obs"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])

        # Standardize the advantage function to have mean=0 and std=1.
        advant_n = np.concatenate([path["advant"] for path in paths])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + 1e-8)

        # Computing baseline function for next iter.
        self.vf.fit(paths)



        feed = {self.obs: obs_n,
                self.action: action_n,
                self.advant: advant_n,
                self.oldaction_dist_mu: action_dist_mu,
                self.oldaction_dist_logstd: action_dist_logstd}

        thprev = self.gf()

        def fisher_vector_product(p):
            feed[self.flat_tangent] = p
            return self.session.run(self.fvp, feed) + p * config.cg_damping


        g = self.session.run(self.pg, feed_dict=feed)
        stepdir = conjugate_gradient(fisher_vector_product, -g)
        shs = (.5 * stepdir.dot(fisher_vector_product(stepdir)) )
        assert shs > 0

        lm = np.sqrt(shs / config.max_kl)
        print(shs)
        print(lm)


        fullstep = stepdir / lm
        neggdotstepdir = -g.dot(stepdir)

        def loss(th):
            self.sff(th)
            return self.session.run(self.losses[0], feed_dict=feed)
        theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
        theta = thprev + fullstep

        self.sff(theta)

        surrafter, kloldnew, entropy = self.session.run(
            self.losses, feed_dict=feed)


        episoderewards = np.array(
            [path["rewards"].sum() for path in paths])
        stats = {}
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Entropy"] = entropy
        # stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
        stats["KL between old and new distribution"] = kloldnew
        stats["Surrogate loss"] = surrafter
        # print ("\n********** Iteration {} ************".format(i))
        for k, v in stats.iteritems():
            print(k + ": " + " " * (40 - len(k)) + str(v))
