# parallel-trpo

A parallel implementation of Trust Region Policy Optimization on environments from OpenAI gym

I'm working towards the ideas at [this openAI research request](https://openai.com/requests-for-research/#parallel-trpo).
The code is based off of [this implementation](https://github.com/ilyasu123/trpo).

How to run:
```
python main.py
```
Parameters:
```
--task: what gym environment to run on
--timesteps_per_batch: how many timesteps for each policy iteration
--max_pathlength: maximum timesteps in one episode
--n_iter: number of iterations
--gamma: discount factor for future rewards_1
--max_kl: maximum KL divergence between new and old policy
--cg_damping: damp on the KL constraint (ratio of original gradient to use)
--num_threads: how many async threads to use
--monitor: whether to monitor progress for publishing results to gym or not
```
