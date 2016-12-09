# parallel-trpo

A parallel implementation of Trust Region Policy Optimization on environments from OpenAI gym

Now includes hyperparaemter adaptation as well! More more info, check [my post on this project](http://kvfrans.com/speeding-up-trpo-through-parallelization-and-parameter-adaptation/).

I'm working towards the ideas at [this openAI research request](https://openai.com/requests-for-research/#parallel-trpo).
The code is based off of [this implementation](https://github.com/ilyasu123/trpo).

I'm currently working together with [Danijar](https://github.com/danijar) to improve the sample efficiency as well.

[Here's a preliminary paper](http://kvfrans.com/static/trpo.pdf) describing the multiple actors setup.

How to run:
```
# This just runs a simple training on Reacher-v1.
python main.py

# For the commands used to recreate results, check trials.txt

```
Parameters:
```
--task: what gym environment to run on
--timesteps_per_batch: how many timesteps for each policy iteration
--n_iter: number of iterations
--gamma: discount factor for future rewards_1
--max_kl: maximum KL divergence between new and old policy
--cg_damping: damp on the KL constraint (ratio of original gradient to use)
--num_threads: how many async threads to use
--monitor: whether to monitor progress for publishing results to gym or not
```
